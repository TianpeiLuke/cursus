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
from typing import Dict, List, Optional, Type, Any

from .models import StepInfo, FileMetadata, StepSearchResult
from .config_discovery import ConfigAutoDiscovery

logger = logging.getLogger(__name__)


class StepCatalog:
    """
    Unified step catalog addressing all validated user stories (US1-US5).
    
    This single class consolidates the functionality of 16+ discovery systems
    while maintaining simple, efficient O(1) lookups through dictionary-based indexing.
    """
    
    def __init__(self, workspace_root: Path):
        """
        Initialize the unified step catalog.
        
        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = workspace_root
        self.config_discovery = ConfigAutoDiscovery(workspace_root)
        self.logger = logging.getLogger(__name__)
        
        # Simple in-memory indexes (US4: Efficient Scaling)
        self._step_index: Dict[str, StepInfo] = {}
        self._component_index: Dict[Path, str] = {}
        self._workspace_steps: Dict[str, List[str]] = {}
        self._index_built = False
        
        # Simple metrics collection
        self.metrics: Dict[str, Any] = {
            'queries': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'index_build_time': 0.0,
            'last_index_build': None
        }
    
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
        List all available steps, optionally filtered by workspace and job_type.
        
        Args:
            workspace_id: Optional workspace filter
            job_type: Optional job type filter
            
        Returns:
            List of step names matching the criteria
        """
        try:
            self._ensure_index_built()
            
            if workspace_id:
                steps = self._workspace_steps.get(workspace_id, [])
            else:
                steps = list(self._step_index.keys())
            
            if job_type:
                # Filter steps by job type
                filtered_steps = []
                for step in steps:
                    if step.endswith(f"_{job_type}") or job_type == "default":
                        filtered_steps.append(step)
                steps = filtered_steps
            
            return steps
            
        except Exception as e:
            self.logger.error(f"Error listing steps for workspace {workspace_id}: {e}")
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
        return self.config_discovery.discover_config_classes(project_id)
    
    def build_complete_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Build complete mapping integrating manual registration with auto-discovery.
        
        This addresses the TODO in the existing build_complete_config_classes() function.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Complete dictionary of config classes (manual + auto-discovered)
        """
        return self.config_discovery.build_complete_config_classes(project_id)
    
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
    
    # Private methods for simple implementation
    def _ensure_index_built(self) -> None:
        """Build index on first access (lazy loading)."""
        if not self._index_built:
            self._build_index()
            self._index_built = True
    
    def _build_index(self) -> None:
        """Simple index building using directory traversal."""
        start_time = time.time()
        
        try:
            # Load registry data first
            try:
                from ..registry.step_names import STEP_NAMES
                
                for step_name, registry_data in STEP_NAMES.items():
                    step_info = StepInfo(
                        step_name=step_name,
                        workspace_id="core",
                        registry_data=registry_data,
                        file_components={}
                    )
                    self._step_index[step_name] = step_info
                    self._workspace_steps.setdefault("core", []).append(step_name)
                
                step_names_dict = STEP_NAMES() if callable(STEP_NAMES) else STEP_NAMES
                self.logger.debug(f"Loaded {len(step_names_dict)} steps from registry")
                
            except ImportError as e:
                self.logger.warning(f"Could not import STEP_NAMES registry: {e}")
            
            # Discover file components across workspaces
            core_steps_dir = self.workspace_root / "src" / "cursus" / "steps"
            if core_steps_dir.exists():
                self._discover_workspace_components("core", core_steps_dir)
            
            # Discover developer workspaces
            dev_projects_dir = self.workspace_root / "development" / "projects"
            if dev_projects_dir.exists():
                for project_dir in dev_projects_dir.iterdir():
                    if project_dir.is_dir():
                        workspace_steps_dir = project_dir / "src" / "cursus_dev" / "steps"
                        if workspace_steps_dir.exists():
                            self._discover_workspace_components(project_dir.name, workspace_steps_dir)
            
            # Record successful build
            build_time = time.time() - start_time
            self.metrics['index_build_time'] = build_time
            self.metrics['last_index_build'] = datetime.now()
            
            self.logger.info(f"Index built successfully in {build_time:.3f}s with {len(self._step_index)} steps")
            
        except Exception as e:
            build_time = time.time() - start_time
            self.logger.error(f"Index build failed after {build_time:.3f}s: {e}")
            # Graceful degradation - use empty index
            self._step_index = {}
            self._component_index = {}
            self._workspace_steps = {}
    
    def _discover_workspace_components(self, workspace_id: str, steps_dir: Path) -> None:
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
        Add component to index with error handling.
        
        Args:
            step_name: Name of the step
            py_file: Path to the component file
            component_type: Type of component
            workspace_id: ID of the workspace
        """
        try:
            # Update or create step info
            if step_name in self._step_index:
                step_info = self._step_index[step_name]
                # Update workspace if this is from a developer workspace
                if workspace_id != "core":
                    step_info.workspace_id = workspace_id
            else:
                step_info = StepInfo(
                    step_name=step_name,
                    workspace_id=workspace_id,
                    registry_data={},
                    file_components={}
                )
                self._step_index[step_name] = step_info
                self._workspace_steps.setdefault(workspace_id, []).append(step_name)
            
            # Add file component
            file_metadata = FileMetadata(
                path=py_file,
                file_type=component_type,
                modified_time=datetime.fromtimestamp(py_file.stat().st_mtime)
            )
            step_info.file_components[component_type] = file_metadata
            self._component_index[py_file] = step_name
            
        except Exception as e:
            self.logger.warning(f"Error adding component {py_file} to index: {e}")
    
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
