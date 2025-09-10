"""
Workspace-Aware Pipeline Testing Specification Builder

Enhanced builder that extends PipelineTestingSpecBuilder with workspace-aware capabilities
for intelligent script discovery across multiple developer workspaces.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from difflib import get_close_matches

from .runtime_spec_builder import PipelineTestingSpecBuilder
from .runtime_models import ScriptExecutionSpec


class WorkspaceAwarePipelineTestingSpecBuilder(PipelineTestingSpecBuilder):
    """
    Enhanced PipelineTestingSpecBuilder with workspace-aware script discovery.
    
    Extends the base builder with:
    - Multi-workspace script discovery via WorkspaceDiscoveryManager
    - Configurable workspace search patterns and depth limits
    - Intelligent priority handling (exact matches > fuzzy matches)
    - Graceful fallback when workspace system unavailable
    - Support for distributed development environments
    """
    
    def __init__(self, test_data_dir: str = "test/integration/runtime", **workspace_config):
        """
        Initialize workspace-aware spec builder.
        
        Args:
            test_data_dir: Root directory for test data and specs
            **workspace_config: Workspace configuration options:
                - workspace_discovery_enabled: Enable workspace-aware discovery (default: True)
                - max_workspace_depth: Maximum workspace search depth (default: 3)
                - workspace_script_patterns: Script directory patterns to search
        """
        super().__init__(test_data_dir)
        
        # Workspace configuration
        self.workspace_discovery_enabled = workspace_config.get('workspace_discovery_enabled', True)
        self.max_workspace_depth = workspace_config.get('max_workspace_depth', 3)
        self.workspace_script_patterns = workspace_config.get('workspace_script_patterns', [
            "scripts/",
            "src/scripts/", 
            "src/cursus/steps/scripts/",
            "validation/scripts/",
            "cursus/steps/scripts/"
        ])
        
        # Cache for workspace discovery results
        self._workspace_cache = {}
        self._workspace_discovery_attempted = False
    
    def _find_actual_script_file(self, canonical_name: str) -> str:
        """
        Enhanced file discovery with workspace-aware capabilities.
        
        Priority order:
        1. Test data scripts (self.scripts_dir) - for testing workspace
        2. Workspace-aware script discovery - across developer workspaces
        3. Core framework scripts (src/cursus/steps/scripts/) - fallback
        
        Args:
            canonical_name: Canonical step name (e.g., "TabularPreprocessing")
            
        Returns:
            Actual script name that exists in one of the script directories
            
        Raises:
            ValueError: If no suitable script file can be found
        """
        # Get expected script name from canonical name
        expected_name = self._canonical_to_script_name(canonical_name)
        
        # Define search directories in priority order
        search_dirs = []
        
        # 1. Test data scripts (highest priority)
        if self.scripts_dir.exists():
            search_dirs.append(("test_data", self.scripts_dir))
        
        # 2. Workspace-aware script discovery (enhanced capability)
        workspace_dirs = self._find_in_workspace(expected_name)
        search_dirs.extend(workspace_dirs)
        
        # 3. Core framework scripts (fallback)
        core_scripts_dir = Path("src/cursus/steps/scripts")
        if core_scripts_dir.exists():
            search_dirs.append(("core", core_scripts_dir))
        
        if not search_dirs:
            raise ValueError("No script directories found (test_data, workspace, or core)")
        
        # Search in priority order
        for location, scripts_dir in search_dirs:
            # Get all Python script files (excluding __init__.py)
            available_scripts = [
                f.stem for f in scripts_dir.glob("*.py") 
                if f.name != "__init__.py"
            ]
            
            # Try exact match first
            if expected_name in available_scripts:
                return expected_name
            
            # Try fuzzy matching if exact match fails
            close_matches = get_close_matches(
                expected_name, 
                available_scripts, 
                n=3, cutoff=0.6
            )
            
            if close_matches:
                best_match = close_matches[0]
                print(f"Warning: Using fuzzy match '{best_match}' for expected '{expected_name}' in {location}")
                return best_match
        
        # Comprehensive error if no matches found
        all_available = []
        for location, scripts_dir in search_dirs:
            scripts = [f.stem for f in scripts_dir.glob("*.py") if f.name != "__init__.py"]
            all_available.extend([f"{script} ({location})" for script in scripts])
        
        raise ValueError(
            f"Cannot find script file for canonical name '{canonical_name}'. "
            f"Expected: '{expected_name}', Available: {all_available}"
        )
    
    def _find_in_workspace(self, script_name: str) -> List[Tuple[str, Path]]:
        """
        Enhanced workspace-aware script discovery using WorkspaceDiscoveryManager.
        
        This method leverages the cursus workspace system to discover scripts across
        multiple developer workspaces, providing intelligent fallback when scripts
        are not found in the immediate test environment.
        
        Args:
            script_name: Expected script name (e.g., "tabular_preprocessing")
            
        Returns:
            List of (location_name, directory_path) tuples for workspace script directories
        """
        workspace_dirs = []
        
        # Check if workspace discovery is enabled
        if not self.workspace_discovery_enabled:
            return workspace_dirs
        
        # Use cached results if available
        if script_name in self._workspace_cache:
            return self._workspace_cache[script_name]
        
        try:
            # Try to import workspace components
            from cursus.workspace.api import WorkspaceAPI
            from cursus.workspace.core import WorkspaceDiscoveryManager
            
            # Initialize workspace discovery manager
            discovery_manager = WorkspaceDiscoveryManager()
            workspace_api = WorkspaceAPI()
            
            # Get available workspaces
            workspaces = discovery_manager.discover_workspaces(
                max_depth=self.max_workspace_depth
            )
            
            # Search each workspace for script directories
            for workspace in workspaces:
                workspace_path = Path(workspace.root_path)
                
                for pattern in self.workspace_script_patterns:
                    script_dir = workspace_path / pattern
                    
                    if script_dir.exists() and script_dir.is_dir():
                        # Check if the expected script exists in this directory
                        script_file = script_dir / f"{script_name}.py"
                        if script_file.exists():
                            location_name = f"workspace_{workspace.name}_{pattern.replace('/', '_').rstrip('_')}"
                            workspace_dirs.append((location_name, script_dir))
                        
                        # Also add directory for fuzzy matching even if exact match not found
                        elif any(f.suffix == '.py' and f.name != '__init__.py' for f in script_dir.iterdir()):
                            location_name = f"workspace_{workspace.name}_{pattern.replace('/', '_').rstrip('_')}_fuzzy"
                            workspace_dirs.append((location_name, script_dir))
            
            # Sort by priority: exact matches first, then fuzzy match directories
            workspace_dirs.sort(key=lambda x: (
                0 if 'fuzzy' not in x[0] else 1,  # Exact matches first
                x[0]  # Then alphabetical
            ))
            
        except ImportError:
            # Workspace system not available, fall back to hardcoded paths
            print("Warning: Workspace system not available, using hardcoded workspace paths")
            workspace_dirs.extend(self._get_fallback_workspace_dirs())
            
        except Exception as e:
            # Log workspace discovery errors but don't fail the entire resolution
            print(f"Warning: Workspace discovery failed: {e}")
            workspace_dirs.extend(self._get_fallback_workspace_dirs())
        
        # Cache results for future use
        self._workspace_cache[script_name] = workspace_dirs
        
        return workspace_dirs
    
    def _get_fallback_workspace_dirs(self) -> List[Tuple[str, Path]]:
        """
        Fallback workspace directories when WorkspaceDiscoveryManager is unavailable.
        
        Returns:
            List of (location_name, directory_path) tuples for common workspace locations
        """
        fallback_dirs = []
        
        # Common workspace patterns
        common_patterns = [
            ("workspace_dev_scripts", Path("../dev_workspace/scripts")),
            ("workspace_project_scripts", Path("../project_workspace/src/scripts")),
            ("workspace_local_scripts", Path("./workspace/scripts")),
            ("workspace_cursus_steps", Path("../cursus/steps/scripts")),
            ("workspace_src_cursus", Path("../src/cursus/steps/scripts")),
        ]
        
        for location_name, dir_path in common_patterns:
            if dir_path.exists() and dir_path.is_dir():
                fallback_dirs.append((location_name, dir_path))
        
        return fallback_dirs
    
    def clear_workspace_cache(self) -> None:
        """Clear the workspace discovery cache to force re-discovery."""
        self._workspace_cache.clear()
        self._workspace_discovery_attempted = False
        print("Workspace discovery cache cleared")
    
    def get_workspace_discovery_status(self) -> Dict[str, Any]:
        """
        Get status information about workspace discovery.
        
        Returns:
            Dictionary with workspace discovery status and statistics
        """
        status = {
            "workspace_discovery_enabled": self.workspace_discovery_enabled,
            "max_workspace_depth": self.max_workspace_depth,
            "workspace_script_patterns": self.workspace_script_patterns,
            "cache_size": len(self._workspace_cache),
            "cached_scripts": list(self._workspace_cache.keys()),
            "discovery_attempted": self._workspace_discovery_attempted
        }
        
        # Try to get workspace system status
        try:
            from cursus.workspace.api import WorkspaceAPI
            from cursus.workspace.core import WorkspaceDiscoveryManager
            
            discovery_manager = WorkspaceDiscoveryManager()
            workspaces = discovery_manager.discover_workspaces(max_depth=self.max_workspace_depth)
            
            status["workspace_system_available"] = True
            status["discovered_workspaces"] = len(workspaces)
            status["workspace_names"] = [ws.name for ws in workspaces]
            
        except ImportError:
            status["workspace_system_available"] = False
            status["error"] = "Workspace system not available (ImportError)"
        except Exception as e:
            status["workspace_system_available"] = False
            status["error"] = f"Workspace discovery error: {str(e)}"
        
        return status
    
    def configure_workspace_discovery(self, **config) -> None:
        """
        Update workspace discovery configuration.
        
        Args:
            **config: Configuration options to update:
                - workspace_discovery_enabled: Enable/disable workspace discovery
                - max_workspace_depth: Maximum search depth
                - workspace_script_patterns: List of script directory patterns
        """
        if 'workspace_discovery_enabled' in config:
            self.workspace_discovery_enabled = config['workspace_discovery_enabled']
        
        if 'max_workspace_depth' in config:
            self.max_workspace_depth = config['max_workspace_depth']
        
        if 'workspace_script_patterns' in config:
            self.workspace_script_patterns = config['workspace_script_patterns']
        
        # Clear cache when configuration changes
        self.clear_workspace_cache()
        
        print(f"Workspace discovery configuration updated: {config}")
    
    def discover_available_scripts(self) -> Dict[str, List[str]]:
        """
        Discover all available scripts across all workspace locations.
        
        Returns:
            Dictionary mapping location names to lists of available script names
        """
        all_scripts = {}
        
        # Test data scripts
        if self.scripts_dir.exists():
            test_scripts = [
                f.stem for f in self.scripts_dir.glob("*.py") 
                if f.name != "__init__.py"
            ]
            if test_scripts:
                all_scripts["test_data"] = test_scripts
        
        # Workspace scripts (use empty script name to get all workspace directories)
        workspace_dirs = self._find_in_workspace("")
        
        for location_name, script_dir in workspace_dirs:
            if script_dir.exists():
                workspace_scripts = [
                    f.stem for f in script_dir.glob("*.py") 
                    if f.name != "__init__.py"
                ]
                if workspace_scripts:
                    all_scripts[location_name] = workspace_scripts
        
        # Core framework scripts
        core_scripts_dir = Path("src/cursus/steps/scripts")
        if core_scripts_dir.exists():
            core_scripts = [
                f.stem for f in core_scripts_dir.glob("*.py") 
                if f.name != "__init__.py"
            ]
            if core_scripts:
                all_scripts["core"] = core_scripts
        
        return all_scripts
    
    def validate_workspace_setup(self) -> Dict[str, Any]:
        """
        Validate workspace setup and provide diagnostic information.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation = {
            "status": "success",
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check workspace discovery system
        try:
            from cursus.workspace.api import WorkspaceAPI
            from cursus.workspace.core import WorkspaceDiscoveryManager
            
            discovery_manager = WorkspaceDiscoveryManager()
            workspaces = discovery_manager.discover_workspaces(max_depth=self.max_workspace_depth)
            
            if not workspaces:
                validation["warnings"].append("No workspaces discovered")
                validation["recommendations"].append("Check workspace configuration and directory structure")
            else:
                validation["discovered_workspaces"] = len(workspaces)
                
        except ImportError:
            validation["errors"].append("Workspace system not available (ImportError)")
            validation["recommendations"].append("Install cursus workspace components")
            validation["status"] = "error"
        except Exception as e:
            validation["errors"].append(f"Workspace discovery failed: {str(e)}")
            validation["status"] = "error"
        
        # Check script directories
        script_locations = self.discover_available_scripts()
        if not script_locations:
            validation["warnings"].append("No script directories found")
            validation["recommendations"].append("Create script directories or check workspace configuration")
        else:
            validation["script_locations"] = len(script_locations)
            validation["total_scripts"] = sum(len(scripts) for scripts in script_locations.values())
        
        # Check configuration
        if not self.workspace_discovery_enabled:
            validation["warnings"].append("Workspace discovery is disabled")
            validation["recommendations"].append("Enable workspace discovery for enhanced script resolution")
        
        if self.max_workspace_depth < 2:
            validation["warnings"].append("Workspace search depth is very shallow")
            validation["recommendations"].append("Consider increasing max_workspace_depth for better discovery")
        
        return validation
