"""
Developer Workspace File Resolver

Extends FlexibleFileResolver to support multi-developer workspace structures.
Provides workspace-aware file discovery for contracts, specs, builders, and scripts.

Architecture:
- Extends existing FlexibleFileResolver capabilities
- Supports developer workspace directory structures
- Maintains backward compatibility with single workspace mode
- Provides workspace isolation and path management

Developer Workspace Structure:
developer_workspaces/
├── developers/
│   ├── developer_1/
│   │   └── src/cursus_dev/steps/
│   │       ├── builders/
│   │       ├── contracts/
│   │       ├── scripts/
│   │       ├── specs/
│   │       └── configs/
│   └── developer_2/
│       └── src/cursus_dev/steps/
│           ├── builders/
│           ├── contracts/
│           ├── scripts/
│           ├── specs/
│           └── configs/
└── shared/
    └── src/cursus_dev/steps/
        ├── builders/
        ├── contracts/
        ├── scripts/
        ├── specs/
        └── configs/
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging

from ...validation.alignment.file_resolver import FlexibleFileResolver


logger = logging.getLogger(__name__)


class DeveloperWorkspaceFileResolver(FlexibleFileResolver):
    """
    Workspace-aware file resolver that extends FlexibleFileResolver
    to support multi-developer workspace structures.
    
    Features:
    - Developer workspace discovery and validation
    - Workspace-specific file resolution with fallback to shared resources
    - Path isolation between developer workspaces
    - Backward compatibility with single workspace mode
    """
    
    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        developer_id: Optional[str] = None,
        enable_shared_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize workspace-aware file resolver.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            developer_id: Specific developer workspace to target
            enable_shared_fallback: Whether to fallback to shared workspace
            **kwargs: Additional arguments passed to FlexibleFileResolver
        """
        # Initialize parent with default paths if not in workspace mode
        if workspace_root is None:
            # Provide default base directories for single workspace mode
            default_base_directories = kwargs.get('base_directories', {
                'contracts': 'src/cursus/steps/contracts',
                'specs': 'src/cursus/steps/specs', 
                'builders': 'src/cursus/steps/builders',
                'configs': 'src/cursus/steps/configs'
            })
            super().__init__(base_directories=default_base_directories)
            self.workspace_mode = False
            self.workspace_root = None
            self.developer_id = None
            self.enable_shared_fallback = False
            return
            
        self.workspace_mode = True
        self.workspace_root = Path(workspace_root)
        self.developer_id = developer_id
        self.enable_shared_fallback = enable_shared_fallback
        
        # Validate workspace structure
        self._validate_workspace_structure()
        
        # Build workspace-specific paths
        workspace_paths = self._build_workspace_paths()

        # Convert workspace paths to base_directories format expected by FlexibleFileResolver
        base_directories = {
            'contracts': workspace_paths['contracts_dir'],
            'specs': workspace_paths['specs_dir'],
            'builders': workspace_paths['builders_dir'],
            'configs': workspace_paths['configs_dir']
        }

        # Initialize parent with base_directories
        super().__init__(base_directories=base_directories, **kwargs)
        
        # Set workspace-specific attributes for direct access
        for key, value in workspace_paths.items():
            setattr(self, key, value)
        
        logger.info(f"Initialized workspace resolver for developer '{developer_id}' "
                   f"at '{workspace_root}'")
    
    def _validate_workspace_structure(self) -> None:
        """Validate that workspace root has expected structure."""
        if not self.workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {self.workspace_root}")
        
        developers_dir = self.workspace_root / "developers"
        shared_dir = self.workspace_root / "shared"
        
        if not developers_dir.exists() and not shared_dir.exists():
            raise ValueError(
                f"Workspace root must contain 'developers' or 'shared' directory: "
                f"{self.workspace_root}"
            )
        
        if self.developer_id:
            dev_workspace = developers_dir / self.developer_id
            if not dev_workspace.exists():
                raise ValueError(
                    f"Developer workspace does not exist: {dev_workspace}"
                )
    
    def _build_workspace_paths(self) -> Dict[str, Any]:
        """Build workspace-specific paths for FlexibleFileResolver."""
        paths = {}
        
        if self.developer_id:
            # Primary paths from developer workspace
            dev_base = (self.workspace_root / "developers" / self.developer_id / 
                       "src" / "cursus_dev" / "steps")
            
            if dev_base.exists():
                paths.update({
                    'contracts_dir': str(dev_base / "contracts"),
                    'specs_dir': str(dev_base / "specs"),
                    'builders_dir': str(dev_base / "builders"),
                    'scripts_dir': str(dev_base / "scripts"),
                    'configs_dir': str(dev_base / "configs"),
                })
        
        # Shared fallback paths
        if self.enable_shared_fallback:
            shared_base = (self.workspace_root / "shared" / "src" / 
                          "cursus_dev" / "steps")
            
            if shared_base.exists():
                # Add shared paths as fallback directories
                shared_paths = {
                    'shared_contracts_dir': str(shared_base / "contracts"),
                    'shared_specs_dir': str(shared_base / "specs"),
                    'shared_builders_dir': str(shared_base / "builders"),
                    'shared_scripts_dir': str(shared_base / "scripts"),
                    'shared_configs_dir': str(shared_base / "configs"),
                }
                paths.update(shared_paths)
        
        return paths
    
    def find_contract_file(self, step_name: str) -> Optional[str]:
        """
        Find contract file with workspace-aware search.
        
        Search order:
        1. Developer workspace contracts
        2. Shared workspace contracts (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            return super().find_contract_file(step_name)
        
        # Try developer workspace first
        result = super().find_contract_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'contracts', step_name, None
            )
            if result:
                return result
        
        return None
    
    def find_spec_file(self, step_name: str) -> Optional[str]:
        """
        Find spec file with workspace-aware search.
        
        Search order:
        1. Developer workspace specs
        2. Shared workspace specs (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            return super().find_spec_file(step_name)
        
        # Try developer workspace first
        result = super().find_spec_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'specs', step_name, None
            )
            if result:
                return result
        
        return None
    
    def find_builder_file(self, step_name: str) -> Optional[str]:
        """
        Find builder file with workspace-aware search.
        
        Search order:
        1. Developer workspace builders
        2. Shared workspace builders (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            return super().find_builder_file(step_name)
        
        # Try developer workspace first
        result = super().find_builder_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'builders', step_name, None
            )
            if result:
                return result
        
        return None
    
    def find_script_file(
        self,
        step_name: str,
        script_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Find script file with workspace-aware search.
        
        Search order:
        1. Developer workspace scripts
        2. Shared workspace scripts (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            # Parent class doesn't have find_script_file, implement basic logic
            return self._find_file_in_directory(
                getattr(self, 'scripts_dir', 'src/cursus/steps/scripts'),
                step_name,
                script_name,
                ['.py']
            )
        
        # Try developer workspace first
        result = self._find_file_in_directory(
            getattr(self, 'scripts_dir', ''),
            step_name,
            script_name,
            ['.py']
        )
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'scripts', step_name, script_name
            )
            if result:
                return result
        
        return None
    
    def find_config_file(
        self,
        step_name: str,
        config_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Find config file with workspace-aware search.
        
        Search order:
        1. Developer workspace configs
        2. Shared workspace configs (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            # Parent class doesn't have find_config_file, implement basic logic
            return self._find_file_in_directory(
                getattr(self, 'configs_dir', 'src/cursus/steps/configs'),
                step_name,
                config_name,
                ['.py']  # Config files are Python files in cursus/steps
            )
        
        # Try developer workspace first using parent class method
        result = super().find_config_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'configs', step_name, config_name
            )
            if result:
                return result
        
        return None
    
    def _find_in_shared_workspace(
        self,
        file_type: str,
        step_name: str,
        file_name: Optional[str] = None
    ) -> Optional[str]:
        """Find file in shared workspace directory."""
        shared_dir_attr = f'shared_{file_type}_dir'
        shared_dir = getattr(self, shared_dir_attr, None)
        
        if not shared_dir or not os.path.exists(shared_dir):
            return None
        
        # Determine file extensions based on type
        extensions = {
            'contracts': ['.py'],
            'specs': ['.json', '.yaml', '.yml'],
            'builders': ['.py'],
            'scripts': ['.py'],
            'configs': ['.json', '.yaml', '.yml']
        }.get(file_type, ['.py', '.json', '.yaml', '.yml'])
        
        return self._find_file_in_directory(
            shared_dir, step_name, file_name, extensions
        )
    
    def _find_file_in_directory(
        self,
        directory: str,
        step_name: str,
        file_name: Optional[str],
        extensions: List[str]
    ) -> Optional[str]:
        """Find file in specified directory with given extensions."""
        if not directory or not os.path.exists(directory):
            return None
        
        # Create a temporary FlexibleFileResolver for the specific directory
        temp_base_dirs = {}
        for ext in extensions:
            if ext == '.py':
                if 'contracts' not in temp_base_dirs:
                    temp_base_dirs['contracts'] = directory
                if 'builders' not in temp_base_dirs:
                    temp_base_dirs['builders'] = directory
            else:
                if 'specs' not in temp_base_dirs:
                    temp_base_dirs['specs'] = directory
                if 'configs' not in temp_base_dirs:
                    temp_base_dirs['configs'] = directory
        
        if temp_base_dirs:
            temp_resolver = FlexibleFileResolver(temp_base_dirs)
            
            # Try different component types based on extensions
            if '.py' in extensions:
                result = temp_resolver.find_contract_file(step_name)
                if result:
                    return result
                result = temp_resolver.find_builder_file(step_name)
                if result:
                    return result
            
            if any(ext in ['.json', '.yaml', '.yml'] for ext in extensions):
                result = temp_resolver.find_spec_file(step_name)
                if result:
                    return result
        
        # Fallback to basic file search
        search_names = []
        if file_name:
            search_names.append(file_name)
        search_names.append(step_name)
        
        for name in search_names:
            for ext in extensions:
                file_path = os.path.join(directory, f"{name}{ext}")
                if os.path.exists(file_path):
                    return file_path
        
        return None
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about current workspace configuration."""
        return {
            'workspace_mode': self.workspace_mode,
            'workspace_root': str(self.workspace_root) if self.workspace_root else None,
            'developer_id': self.developer_id,
            'enable_shared_fallback': self.enable_shared_fallback,
            'developer_workspace_exists': (
                self.workspace_root and self.developer_id and
                (self.workspace_root / "developers" / self.developer_id).exists()
            ) if self.workspace_mode else False,
            'shared_workspace_exists': (
                self.workspace_root and
                (self.workspace_root / "shared").exists()
            ) if self.workspace_mode else False
        }
    
    def list_available_developers(self) -> List[str]:
        """List all available developer workspaces."""
        if not self.workspace_mode or not self.workspace_root:
            return []
        
        developers_dir = self.workspace_root / "developers"
        if not developers_dir.exists():
            return []
        
        developers = []
        for item in developers_dir.iterdir():
            if item.is_dir():
                # Check if it has the expected structure
                cursus_dev_dir = item / "src" / "cursus_dev" / "steps"
                if cursus_dev_dir.exists():
                    developers.append(item.name)
        
        return sorted(developers)
    
    def switch_developer(self, developer_id: str) -> None:
        """Switch to a different developer workspace."""
        if not self.workspace_mode:
            raise ValueError("Not in workspace mode")
        
        if developer_id not in self.list_available_developers():
            raise ValueError(f"Developer workspace not found: {developer_id}")
        
        self.developer_id = developer_id
        
        # Rebuild paths for new developer
        workspace_paths = self._build_workspace_paths()
        
        # Update base directories for FlexibleFileResolver
        new_base_directories = {
            'contracts': workspace_paths['contracts_dir'],
            'specs': workspace_paths['specs_dir'],
            'builders': workspace_paths['builders_dir'],
            'configs': workspace_paths['configs_dir']
        }
        
        # Update parent class base directories and refresh cache
        self.base_dirs = {k: Path(v) for k, v in new_base_directories.items()}
        self._discover_all_files()
        
        # Update instance attributes
        for key, value in workspace_paths.items():
            setattr(self, key, value)
        
        logger.info(f"Switched to developer workspace: {developer_id}")
