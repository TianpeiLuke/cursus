---
tags:
  - design
  - step_catalog
  - search_management
  - pypi_packaging
  - workspace_aware
keywords:
  - unified step catalog
  - search space management
  - PyPI packaging
  - workspace-aware system
  - flexible source structure
  - component discovery
  - separation of concerns
  - package discovery
  - workspace discovery
topics:
  - step catalog search architecture
  - PyPI compatibility design
  - workspace-aware component discovery
language: python
date of note: 2025-09-19
---

# Unified Step Catalog System Search Space Management Design

## Executive Summary

This document presents a streamlined design for search space management in the unified step catalog system, addressing PyPI packaging compatibility, flexible source structure support, and workspace-aware functionality. Following code redundancy reduction principles, this design maintains the step catalog's core functionality while extending search capabilities through minimal, focused mechanisms.

### Key Design Principles

1. **Package-Relative Discovery**: Use relative paths within the package structure for core component discovery
2. **Optional Workspace Extension**: Support additional workspace directories without requiring them
3. **Deployment Agnostic**: Work identically across PyPI installations, source installations, and submodule integrations
4. **Minimal Complexity**: Avoid over-engineering while delivering necessary functionality
5. **Backward Compatibility**: Maintain existing step catalog functionality and APIs
6. **Clear Separation of Concerns**: The system autonomously discovers all components within its own package boundaries, while users must explicitly specify their separate project directories for workspace-aware functionality

### Strategic Impact

- **Universal Deployment**: Single codebase works across all deployment scenarios
- **Flexible Integration**: Support for cursus as submodule or standalone package
- **Enhanced Discovery**: Extended search capabilities without breaking existing functionality
- **Reduced Complexity**: Simple, focused implementation avoiding over-engineering

## Current System Issues

### Problem Analysis

The current unified step catalog system has a fundamental issue with workspace_root parameter:

#### **Inconsistent Path Assumptions**

```python
# CURRENT PROBLEMATIC APPROACH in StepCatalog
if workspace_root is None:
    catalog_dir = Path(__file__).parent  # src/cursus/step_catalog/
    workspace_root = catalog_dir.parent / 'steps'  # ASSUMES src/cursus/steps ❌

# But usage expects project root:
core_steps_dir = workspace_root_path / "src" / "cursus" / "steps"  # ❌ Double nested
dev_projects_dir = workspace_root_path / "development" / "projects"  # ❌ Wrong location
```

#### **Deployment-Specific Failures**

**PyPI Installation Context**:
```
site-packages/cursus/
├── steps/configs/           # Package components
└── step_catalog/           # Step catalog system

# User's separate project directory:
/user/project/
├── development/projects/   # User workspaces
└── my_pipeline.py         # User code
```

**Current system fails because**:
- Looks for `site-packages/cursus/src/cursus/steps/` (doesn't exist)
- Cannot find user workspaces in separate directory
- Conflates package structure with user workspace structure

## Proposed Search Space Management Architecture

### Core Design Concept

Following redundancy reduction principles, the new architecture uses a simple two-domain approach:

1. **Package Search Space**: Components within the cursus package (always available via relative paths)
2. **Workspace Search Space**: User-defined workspace directories (optional, explicitly provided)

### Simplified Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    StepCatalog (Unified)                        │
├─────────────────────────────────────────────────────────────────┤
│  Package Discovery          │  Workspace Discovery (Optional)   │
│  • Autonomous operation     │  • User-specified directories     │
│  • Relative paths           │  • Explicit configuration         │
│  • Always works             │  • Project-specific components    │
│  • Core components          │  • Requires user input            │
└─────────────────────────────────────────────────────────────────┘
```

### Separation of Concerns Architecture

The design enforces clear boundaries between system responsibilities and user responsibilities:

#### **System Responsibilities (Autonomous)**
- **Package Component Discovery**: The system automatically discovers all components within its own package boundaries using relative path navigation
- **Deployment Adaptation**: Automatically adapts to different deployment scenarios (PyPI, source, submodule) without user intervention
- **Core Functionality**: Provides complete step catalog functionality using only package-internal components

#### **User Responsibilities (Explicit Configuration)**
- **Workspace Directory Specification**: Users must explicitly provide paths to their separate project directories for workspace-aware functionality
- **Project Structure Compliance**: User workspace directories must follow the expected `development/projects/` structure
- **Workspace Management**: Users are responsible for organizing and maintaining their project workspace directories

This separation ensures that:
1. **The system never assumes or searches outside its package boundaries**
2. **Users maintain full control over their project directory locations**
3. **No implicit dependencies exist between system and user directory structures**
4. **Clear interface contracts exist between system and user responsibilities**

### Implementation Design

#### **Updated StepCatalog Class**

```python
class StepCatalog:
    """
    Unified step catalog with flexible search space management.
    
    Maintains backward compatibility while supporting flexible deployment scenarios.
    """
    
    def __init__(self, workspace_dirs: Optional[Union[Path, List[Path]]] = None):
        """
        Initialize the unified step catalog with optional workspace directories.
        
        Args:
            workspace_dirs: Optional workspace directory(ies) for workspace-aware discovery.
                           Can be a single Path or list of Paths.
                           Each should contain development/projects/ structure.
                           If None, only discovers package components.
        
        Examples:
            # Package-only discovery (works in all deployment scenarios)
            catalog = StepCatalog()
            
            # Single workspace directory
            catalog = StepCatalog(workspace_dirs=Path("/path/to/workspace"))
            
            # Multiple workspace directories
            catalog = StepCatalog(workspace_dirs=[
                Path("/workspace1"), Path("/workspace2")
            ])
        """
        # Find package root using relative path (deployment agnostic)
        self.package_root = self._find_package_root()
        
        # Normalize workspace_dirs to list
        if workspace_dirs is None:
            self.workspace_dirs = []
        elif isinstance(workspace_dirs, Path):
            self.workspace_dirs = [workspace_dirs]
        else:
            self.workspace_dirs = list(workspace_dirs)
        
        # Initialize config discovery with both search spaces
        self.config_discovery = ConfigAutoDiscovery(self.package_root, self.workspace_dirs)
        
        # Maintain existing attributes for backward compatibility
        self.logger = logging.getLogger(__name__)
        
        # Existing functionality (unchanged)
        self._step_index: Dict[str, StepInfo] = {}
        self._component_index: Dict[Path, str] = {}
        self._workspace_steps: Dict[str, List[str]] = {}
        self._index_built = False
        
        # Existing caches and metrics (unchanged)
        self._framework_cache: Dict[str, str] = {}
        self._validation_metadata_cache: Dict[str, Any] = {}
        self._builder_class_cache: Dict[str, Type] = {}
        self.metrics: Dict[str, Any] = {
            'queries': 0, 'errors': 0, 'avg_response_time': 0.0,
            'index_build_time': 0.0, 'last_index_build': None
        }
    
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
                
                # Look for development/projects/ structure
                dev_projects_dir = workspace_path / "development" / "projects"
                if dev_projects_dir.exists():
                    for project_dir in dev_projects_dir.iterdir():
                        if project_dir.is_dir():
                            workspace_steps_dir = project_dir / "src" / "cursus_dev" / "steps"
                            if workspace_steps_dir.exists():
                                self._discover_workspace_components_in_dir(project_dir.name, workspace_steps_dir)
                else:
                    self.logger.warning(f"Workspace directory missing development/projects structure: {workspace_path}")
                    
            except Exception as e:
                self.logger.error(f"Error discovering workspace components in {workspace_dir}: {e}")
    
    # All existing methods remain unchanged for backward compatibility
    # (get_step_info, find_step_by_component, list_available_steps, etc.)
```

#### **Updated ConfigAutoDiscovery**

```python
class ConfigAutoDiscovery:
    """Configuration class auto-discovery using dual search space approach."""
    
    def __init__(self, package_root: Path, workspace_dirs: List[Path]):
        """
        Initialize with package root and optional workspace directories.
        
        Args:
            package_root: Root of the cursus package
            workspace_dirs: List of workspace directories to search
        """
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs
        self.logger = logging.getLogger(__name__)
    
    def discover_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover configuration classes from package and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        discovered_classes = {}
        
        # Always scan package core configs
        core_config_dir = self.package_root / "steps" / "configs"
        if core_config_dir.exists():
            try:
                core_classes = self._scan_config_directory(core_config_dir)
                discovered_classes.update(core_classes)
                self.logger.info(f"Discovered {len(core_classes)} core config classes")
            except Exception as e:
                self.logger.error(f"Error scanning core config directory: {e}")
        
        # Scan workspace configs if workspace directories provided
        if self.workspace_dirs:
            for workspace_dir in self.workspace_dirs:
                try:
                    workspace_classes = self._discover_workspace_configs(workspace_dir, project_id)
                    # Workspace configs override core configs with same names
                    discovered_classes.update(workspace_classes)
                except Exception as e:
                    self.logger.error(f"Error scanning workspace config directory {workspace_dir}: {e}")
        
        return discovered_classes
    
    def _discover_workspace_configs(self, workspace_dir: Path, project_id: Optional[str] = None) -> Dict[str, Type]:
        """Discover config classes in a workspace directory."""
        discovered = {}
        projects_dir = workspace_dir / "development" / "projects"
        
        if not projects_dir.exists():
            return discovered
        
        if project_id:
            # Search specific project
            project_dir = projects_dir / project_id
            if project_dir.exists():
                config_dir = project_dir / "src" / "cursus_dev" / "steps" / "configs"
                if config_dir.exists():
                    discovered.update(self._scan_config_directory(config_dir))
        else:
            # Search all projects
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    config_dir = project_dir / "src" / "cursus_dev" / "steps" / "configs"
                    if config_dir.exists():
                        discovered.update(self._scan_config_directory(config_dir))
        
        return discovered
    
    def build_complete_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Build complete mapping integrating manual registration with auto-discovery.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Complete dictionary of config classes (manual + auto-discovered)
        """
        try:
            # Import ConfigClassStore if available
            from ..core.config_fields.config_class_store import ConfigClassStore
            
            # Start with manually registered classes (highest priority)
            config_classes = ConfigClassStore.get_all_classes()
            self.logger.debug(f"Found {len(config_classes)} manually registered config classes")
            
            # Add auto-discovered config classes
            discovered_config_classes = self.discover_config_classes(project_id)
            config_added_count = 0
            
            for class_name, class_type in discovered_config_classes.items():
                if class_name not in config_classes:
                    config_classes[class_name] = class_type
                    # Register for consistency
                    try:
                        ConfigClassStore.register(class_type)
                        config_added_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to register auto-discovered config class {class_name}: {e}")
            
            # Add auto-discovered hyperparameter classes
            discovered_hyperparam_classes = self.discover_hyperparameter_classes(project_id)
            hyperparam_added_count = 0
            
            for class_name, class_type in discovered_hyperparam_classes.items():
                if class_name not in config_classes:
                    config_classes[class_name] = class_type
                    hyperparam_added_count += 1
            
            self.logger.info(f"Built complete config classes: {len(config_classes)} total "
                           f"({config_added_count} config + {hyperparam_added_count} hyperparameter auto-discovered)")
            return config_classes
            
        except ImportError as e:
            self.logger.error(f"Failed to import ConfigClassStore: {e}")
            # Fallback to just auto-discovery
            config_classes = self.discover_config_classes(project_id)
            hyperparam_classes = self.discover_hyperparameter_classes(project_id)
            config_classes.update(hyperparam_classes)
            return config_classes
    
    # Existing methods remain unchanged (_scan_config_directory, etc.)
```

## Deployment Scenario Support

### 1. PyPI Installation Scenario

#### **Package Structure**
```
site-packages/cursus/
├── steps/configs/           # Core config classes
└── step_catalog/           # Step catalog system
```

#### **User Project Structure**
```
/user/project/
├── development/projects/   # User workspaces
│   ├── project_alpha/
│   └── project_beta/
└── my_pipeline.py         # User code
```

#### **Usage Pattern**
```python
# Package-only discovery (works automatically)
catalog = StepCatalog()
config_classes = catalog.build_complete_config_classes()

# Workspace-aware discovery (user provides workspace location)
catalog = StepCatalog(workspace_dirs=Path("/user/project"))
config_classes = catalog.build_complete_config_classes(project_id="project_alpha")
```

### 2. Source Installation Scenario

#### **Repository Structure**
```
cursus_repo/
├── src/cursus/
│   ├── steps/configs/      # Core config classes
│   └── step_catalog/       # Step catalog system
└── development/projects/   # Development workspaces
```

#### **Usage Pattern**
```python
# Package discovery works via relative paths
catalog = StepCatalog()

# Workspace discovery with repository structure
catalog = StepCatalog(workspace_dirs=Path("cursus_repo"))
```

### 3. Submodule Integration Scenario

#### **Parent Project Structure**
```
parent_project/
├── external/cursus/        # Cursus as submodule
│   └── steps/configs/      # Core config classes
└── workspaces/            # Parent project workspaces
    └── development/projects/
```

#### **Usage Pattern**
```python
# Package discovery works regardless of parent structure
catalog = StepCatalog()

# Workspace discovery with custom workspace location
catalog = StepCatalog(workspace_dirs=Path("parent_project/workspaces"))
```

## Integration with Existing Systems

### Updated build_complete_config_classes

```python
def build_complete_config_classes(project_id: Optional[str] = None, 
                                workspace_dirs: Optional[Union[Path, List[Path]]] = None) -> Dict[str, Type[BaseModel]]:
    """
    Build complete config classes using flexible search space management.
    
    Args:
        project_id: Optional project ID for workspace-specific discovery
        workspace_dirs: Optional workspace directories for workspace-aware discovery
        
    Returns:
        Dictionary mapping class names to class types
    """
    try:
        from ...step_catalog import StepCatalog
        
        # Create step catalog with flexible search space
        catalog = StepCatalog(workspace_dirs=workspace_dirs)
        
        # Use step catalog's enhanced discovery
        discovered_classes = catalog.build_complete_config_classes(project_id)
        
        logger.info(f"Successfully discovered {len(discovered_classes)} config classes")
        return discovered_classes
        
    except Exception as e:
        logger.error(f"Error in step catalog discovery: {e}")
        # Fallback to legacy implementation
        return _legacy_build_complete_config_classes()
```

## Implementation Strategy

### Phase 1: Core Implementation (Week 1)

#### **Day 1-2: Package Root Detection**
- Implement `_find_package_root()` method with relative path navigation
- Test across PyPI, source, and submodule deployment scenarios
- Ensure backward compatibility with existing functionality

#### **Day 3-4: Workspace Directory Support**
- Add optional `workspace_dirs` parameter to StepCatalog constructor
- Implement workspace component discovery
- Test with multiple workspace directories

#### **Day 5: Integration and Testing**
- Update `build_complete_config_classes()` function
- Update ConfigAutoDiscovery with dual search space
- Comprehensive testing across deployment scenarios

### Phase 2: Validation and Documentation (Week 2)

#### **Day 1-2: Comprehensive Testing**
- Test all deployment scenarios (PyPI, source, submodule)
- Validate backward compatibility
- Performance testing and optimization

#### **Day 3-4: Documentation and Examples**
- Update API documentation
- Create usage examples for different scenarios
- Update integration guides

#### **Day 5: Final Integration**
- Integration with config field management system
- Final validation and deployment preparation

## Separation of Concerns Principle Validation

### Design Compliance Verification

The design strictly enforces the separation of concerns principle through the following mechanisms:

#### **System Autonomy (Package Boundary Enforcement)**

```python
def _find_package_root(self) -> Path:
    """
    System autonomously finds its own package root without external dependencies.
    NEVER searches outside package boundaries.
    """
    current_file = Path(__file__)
    current_dir = current_file.parent
    while current_dir.name != 'cursus' and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    # ONLY returns paths within cursus package
    return current_dir if current_dir.name == 'cursus' else current_file.parent.parent

def _discover_package_components(self) -> None:
    """
    System autonomously discovers components ONLY within its package.
    No assumptions about external directory structures.
    """
    core_steps_dir = self.package_root / "steps"  # ONLY package-internal paths
    if core_steps_dir.exists():
        self._discover_workspace_components_in_dir("core", core_steps_dir)
```

#### **User Explicit Configuration (Workspace Boundary Enforcement)**

```python
def __init__(self, workspace_dirs: Optional[Union[Path, List[Path]]] = None):
    """
    Users MUST explicitly provide workspace directories.
    System makes NO assumptions about user project locations.
    """
    # System handles its own package discovery autonomously
    self.package_root = self._find_package_root()
    
    # User workspace discovery ONLY if explicitly provided
    if workspace_dirs is None:
        self.workspace_dirs = []  # NO implicit workspace discovery
    else:
        self.workspace_dirs = self._normalize_user_provided_dirs(workspace_dirs)

def _discover_workspace_components(self) -> None:
    """
    Workspace discovery ONLY operates on user-provided directories.
    System validates user-provided structure but makes no assumptions.
    """
    for workspace_dir in self.workspace_dirs:  # ONLY user-provided directories
        workspace_path = Path(workspace_dir)
        if not workspace_path.exists():
            self.logger.warning(f"User-provided workspace directory does not exist: {workspace_path}")
            continue
        
        # Validate expected user structure
        dev_projects_dir = workspace_path / "development" / "projects"
        if not dev_projects_dir.exists():
            self.logger.warning(f"User workspace missing expected structure: {workspace_path}")
```

#### **Interface Contract Enforcement**

```python
# SYSTEM RESPONSIBILITY: Always works without user input
catalog = StepCatalog()  # System autonomously discovers package components
config_classes = catalog.build_complete_config_classes()  # Works with package-only

# USER RESPONSIBILITY: Must provide workspace directories for extended functionality
catalog = StepCatalog(workspace_dirs=Path("/user/must/specify/this/path"))
config_classes = catalog.build_complete_config_classes(project_id="user_project")
```

### Principle Fulfillment Checklist

#### ✅ **System Autonomy Requirements Met**
- [x] System discovers all package components without external input
- [x] System adapts to deployment scenarios (PyPI, source, submodule) automatically
- [x] System never searches outside its package boundaries
- [x] System provides complete functionality using only package components
- [x] System makes no assumptions about user directory structures

#### ✅ **User Explicit Configuration Requirements Met**
- [x] Users must explicitly provide workspace directory paths
- [x] Users are responsible for workspace directory structure compliance
- [x] Users maintain full control over their project organization
- [x] Users receive clear feedback when their directories don't meet expectations
- [x] System provides no implicit workspace discovery mechanisms

#### ✅ **Interface Contract Requirements Met**
- [x] Clear separation between system-managed and user-managed components
- [x] Explicit parameters for user-provided configuration
- [x] No hidden dependencies between system and user directory structures
- [x] Backward compatibility maintained for existing system-only usage
- [x] Optional workspace functionality doesn't affect core system operation

## Benefits and Quality Assurance

### Redundancy Reduction Achieved

Following the code redundancy evaluation guide principles:

1. **Avoided Over-Engineering**: No complex search space managers or abstract interfaces
2. **Focused Implementation**: Simple dual-space approach instead of complex multi-layer architecture
3. **Minimal Complexity**: Single class handles both package and workspace discovery
4. **Essential Functionality**: Only implements necessary features for actual requirements

### Quality Preservation

1. **Backward Compatibility**: All existing APIs continue to work unchanged
2. **Performance**: Minimal overhead for package-only usage
3. **Reliability**: Simple, testable implementation with clear error handling
4. **Maintainability**: Straightforward code that's easy to understand and modify

### Expected Redundancy Level

- **Target**: 15-20% redundancy (good efficiency range)
- **Avoided**: 35%+ redundancy that would indicate over-engineering
- **Focus**: Essential functionality with justified architectural patterns

## Workspace-Aware System Compliance Analysis

### Current Violations of Search Space Separation Principle

The existing workspace-aware system under `src/cursus/workspace/` has several violations of our search space separation principle that need to be addressed:

#### **Violation 1: Mixed Search Space Assumptions**

**Location**: `src/cursus/workspace/api.py`
```python
# LINE 47-48: Hardcoded assumption about user workspace structure
def __init__(self, base_path: Optional[Union[str, Path]] = None):
    self.base_path = Path(base_path) if base_path else Path("development")
```

**Problem**: Assumes user workspace is always under "development" directory, violating user explicit configuration principle.

**Location**: `src/cursus/workspace/core/manager.py`
```python
# LINE 89-95: Conflates package and user workspace responsibilities
def __init__(self, workspace_root: Optional[Union[str, Path]] = None, ...):
    self.workspace_root = Path(workspace_root) if workspace_root else None
    # ... 
    # Auto-discover workspaces if requested
    if auto_discover and self.workspace_root:
        self.discover_workspaces()  # VIOLATION: Assumes workspace structure
```

**Problem**: System automatically discovers workspaces without user explicit configuration, violating separation principle.

#### **Violation 2: Hardcoded Path Structure Enforcement**

**Location**: `src/cursus/step_catalog/adapters/workspace_discovery.py`
```python
# LINE 67-85: Hardcoded workspace structure assumptions
def discover_workspaces(self, workspace_root: Path) -> Dict[str, Any]:
    # Discover developer workspaces
    developers_dir = workspace_root / "developers"  # VIOLATION: Hardcoded structure
    if developers_dir.exists():
        # ...
    
    # Discover shared workspace  
    shared_dir = workspace_root / "shared"  # VIOLATION: Hardcoded structure
    if shared_dir.exists():
        # ...
```

**Problem**: Enforces rigid directory structure instead of allowing user flexibility in workspace organization.

**Location**: `src/cursus/step_catalog/adapters/workspace_discovery.py`
```python
# LINE 102-125: Hardcoded component path assumptions
def _count_workspace_components(self, workspace_path: Path) -> int:
    cursus_dev_path = workspace_path / "src" / "cursus_dev" / "steps"  # VIOLATION
    
    if cursus_dev_path.exists():
        # Count builders
        builders_path = cursus_dev_path / "builders"      # VIOLATION
        # Count configs  
        configs_path = cursus_dev_path / "configs"        # VIOLATION
        # Count contracts
        contracts_path = cursus_dev_path / "contracts"    # VIOLATION
        # Count specs
        specs_path = cursus_dev_path / "specs"           # VIOLATION
        # Count scripts
        scripts_path = cursus_dev_path / "scripts"       # VIOLATION
```

**Problem**: Hardcodes specific directory structure (`src/cursus_dev/steps/`) instead of using flexible discovery.

#### **Violation 3: Missing Package Component Discovery**

**Location**: `src/cursus/step_catalog/adapters/workspace_discovery.py`
```python
# LINE 127-155: Only discovers user workspace components, ignores package components
def discover_components(self, workspace_ids: Optional[List[str]] = None, developer_id: Optional[str] = None):
    # Check if workspace root is configured
    if not self.workspace_manager or not hasattr(self.workspace_manager, 'workspace_root'):
        return {"error": "No workspace root configured"}  # VIOLATION: No package fallback
    
    # Only discover components if we have specific workspace constraints
    if self.catalog and (workspace_ids or developer_id):
        # ... only workspace discovery, no package discovery
```

**Problem**: System cannot discover package components autonomously - requires user workspace configuration.

#### **Violation 4: Duplicated Discovery Logic**

**Location**: `src/cursus/step_catalog/adapters/workspace_discovery.py`
```python
# LINE 102-125: Custom discovery logic instead of using StepCatalog
def _count_workspace_components(self, workspace_path: Path) -> int:
    # Custom file counting logic that duplicates StepCatalog functionality
    component_count = 0
    # ... manual file counting instead of using StepCatalog discovery
```

**Problem**: Reimplements component discovery instead of delegating to unified StepCatalog system.

#### **Violation 5: Workspace Root Path Confusion**

**Location**: `src/cursus/workspace/core/manager.py`
```python
# LINE 200-210: Conflates workspace_root with different meanings
def discover_workspaces(self, workspace_root: Optional[Union[str, Path]] = None):
    if workspace_root:
        self.workspace_root = Path(workspace_root)  # VIOLATION: Overrides instance variable
    
    if not self.workspace_root or not self.workspace_root.exists():
        raise ValueError(f"Workspace root does not exist: {self.workspace_root}")
```

**Problem**: Uses `workspace_root` parameter inconsistently - sometimes means user workspace directory, sometimes means system root.

### **Required Fixes for Compliance**

#### **Fix 1: Implement Dual Search Space Architecture**

**Files to Update**:
- `src/cursus/workspace/api.py`
- `src/cursus/workspace/core/manager.py`

**Changes Required**:
```python
# Replace hardcoded base_path with explicit workspace_dirs
class WorkspaceAPI:
    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        # PACKAGE SEARCH SPACE (Autonomous)
        self.package_root = self._find_package_root()
        
        # USER WORKSPACE SEARCH SPACE (Explicit)
        self.workspace_dirs = workspace_dirs or []
        
        # Use StepCatalog with dual search space
        self.catalog = StepCatalog(workspace_dirs=self.workspace_dirs)
```

#### **Fix 2: Remove Hardcoded Path Assumptions**

**Files to Update**:
- `src/cursus/step_catalog/adapters/workspace_discovery.py`

**Changes Required**:
```python
# Replace hardcoded structure assumptions with flexible discovery
def discover_workspaces(self, workspace_dirs: List[Path]) -> Dict[str, Any]:
    # Let StepCatalog handle discovery with user-provided directories
    catalog = StepCatalog(workspace_dirs=workspace_dirs)
    return catalog.get_search_space_info()
```

#### **Fix 3: Add Package Component Discovery**

**Files to Update**:
- `src/cursus/step_catalog/adapters/workspace_discovery.py`

**Changes Required**:
```python
def discover_components(self, workspace_ids: Optional[List[str]] = None):
    # ALWAYS discover package components (autonomous)
    catalog = StepCatalog()  # Package-only discovery
    package_components = catalog.discover_all_components()
    
    # OPTIONALLY discover workspace components (explicit)
    if self.workspace_dirs:
        workspace_catalog = StepCatalog(workspace_dirs=self.workspace_dirs)
        workspace_components = workspace_catalog.discover_all_components(workspace_ids)
        package_components.update(workspace_components)
    
    return package_components
```

#### **Fix 4: Delegate to StepCatalog**

**Files to Update**:
- `src/cursus/step_catalog/adapters/workspace_discovery.py`

**Changes Required**:
```python
# Remove custom discovery logic, delegate to StepCatalog
def _count_workspace_components(self, workspace_path: Path) -> int:
    catalog = StepCatalog(workspace_dirs=[workspace_path])
    components = catalog.discover_all_components()
    return len(components)
```

#### **Fix 5: Consistent Parameter Naming**

**Files to Update**:
- `src/cursus/workspace/core/manager.py`
- `src/cursus/workspace/api.py`

**Changes Required**:
```python
# Replace confusing workspace_root with clear workspace_dirs
def discover_workspaces(self, workspace_dirs: List[Path]) -> Dict[str, Any]:
    # Clear separation: workspace_dirs are user-provided directories
    catalog = StepCatalog(workspace_dirs=workspace_dirs)
    return catalog.get_search_space_info()
```

### **Implementation Priority**

1. **High Priority**: Fix dual search space architecture in `WorkspaceAPI` and `WorkspaceManager`
2. **High Priority**: Remove hardcoded path assumptions in `workspace_discovery.py`
3. **Medium Priority**: Add package component discovery capability
4. **Medium Priority**: Delegate discovery logic to StepCatalog
5. **Low Priority**: Consistent parameter naming across workspace system

These fixes will ensure the workspace-aware system fully complies with our search space separation principle and integrates properly with the unified step catalog design.

## Conclusion

This design provides a streamlined solution for unified step catalog search space management that:

1. **Solves Real Problems**: Addresses actual PyPI packaging and workspace-aware requirements
2. **Avoids Over-Engineering**: Simple, focused implementation without unnecessary complexity
3. **Maintains Quality**: Preserves existing functionality while adding essential capabilities
4. **Supports All Scenarios**: Works across PyPI, source, and submodule deployments
5. **Ensures Consistency**: Provides a framework for fixing workspace-aware system violations

The implementation follows code redundancy reduction principles by delivering necessary functionality through minimal, well-focused mechanisms rather than complex architectural abstractions. The identified violations in the workspace-aware system provide a clear roadmap for achieving full compliance with the separation of concerns principle.

## References

### **Design Documents (slipbox/1_design/)**

#### **Core Step Catalog Architecture**
- **[Unified Step Catalog Config Field Management Refactoring Design](./unified_step_catalog_config_field_management_refactoring_design.md)** - Comprehensive refactoring design for integrating config field management with step catalog system
- **[Config Field Categorization Consolidated](./config_field_categorization_consolidated.md)** - Sophisticated field categorization architecture and three-tier design principles
- **[Config Field Manager Refactoring](./config_field_manager_refactoring.md)** - Registry refactoring and single source of truth principles for config management

#### **Configuration Management Architecture**
- **[Config Manager Three-Tier Implementation](./config_manager_three_tier_implementation.md)** - Three-tier field classification and property-based derivation system
- **[Config Tiered Design](./config_tiered_design.md)** - Tiered configuration architecture principles and implementation patterns
- **[Config Registry](./config_registry.md)** - Configuration class registration system and management patterns
- **[Config Merger](./config_merger.md)** - Configuration merging strategies and field organization principles

#### **System Architecture and Quality**
- **[Code Redundancy Evaluation Guide](./code_redundancy_evaluation_guide.md)** - Principles for redundancy reduction and code quality optimization
- **[Circular Reference Tracker](./circular_reference_tracker.md)** - Circular reference detection and handling in configuration systems
- **[Type-Aware Config Serializer](./type_aware_config_serializer.md)** - Advanced serialization with type preservation and complex object handling

#### **Workspace and Discovery Systems**
- **[Config Class Auto Discovery Design](./config_class_auto_discovery_design.md)** - AST-based configuration class discovery mechanisms
- **[Contract Discovery Manager Design](./contract_discovery_manager_design.md)** - Component contract discovery and validation systems
- **[Adaptive Configuration Management System Revised](./adaptive_configuration_management_system_revised.md)** - Adaptive configuration management patterns

### **Analysis Documents (slipbox/4_analysis/)**

#### **System Analysis and Evaluation**
- **[Config Field Management System Analysis](../4_analysis/config_field_management_system_analysis.md)** - Comprehensive analysis of current config field management issues, redundancy evaluation, and improvement recommendations
- **[Step Catalog System Integration Analysis](../4_analysis/step_catalog_system_integration_analysis.md)** - Integration analysis between step catalog and existing systems, compatibility assessment

#### **Performance and Coverage Analysis**
- **[Legacy System Coverage Analysis](../4_analysis/legacy_system_coverage_analysis.md)** - Analysis of legacy systems including config discovery failures and coverage gaps
- **[Code Quality Assessment](../4_analysis/code_quality_assessment.md)** - Code quality metrics and improvement recommendations across the system

### **Implementation Planning (slipbox/2_project_planning/)**

#### **Primary Implementation Plans**
- **[Search Space Management Improvement Plan](../2_project_planning/2025-09-19_search_space_management_improvement_plan.md)** - Comprehensive 3-week implementation plan for improving search space management across unified step catalog and workspace-aware systems
- **[Config Field Management System Refactoring Implementation Plan](../2_project_planning/2025-09-19_config_field_management_system_refactoring_implementation_plan.md)** - Parallel implementation plan for config field management system refactoring and integration

#### **Step Catalog System Implementation**
- **[Unified Step Catalog System Implementation Plan](../2_project_planning/unified_step_catalog_system_implementation_plan.md)** - Original comprehensive implementation strategy and timeline for unified step catalog system
- **[Unified Step Catalog Migration Guide](../2_project_planning/unified_step_catalog_migration_guide.md)** - Migration procedures and integration patterns for transitioning to unified step catalog

#### **System Integration and Migration**
- **[System Integration Timeline](../2_project_planning/system_integration_timeline.md)** - Coordinated timeline for integrating multiple system improvements and refactoring efforts
- **[Workspace-Aware System Migration Plan](../2_project_planning/workspace_aware_system_migration_plan.md)** - Specific migration plan for workspace-aware system compliance with separation of concerns

### **Related Technical Documentation**

#### **Developer Guides and Standards**
- **[Documentation YAML Frontmatter Standard](./documentation_yaml_frontmatter_standard.md)** - Standardized YAML frontmatter format used across all documentation
- **[API Reference Documentation Style Guide](./api_reference_documentation_style_guide.md)** - Style guide for consistent API documentation across the system

#### **Validation and Testing Framework**
- **[Alignment Validation Data Structures](./alignment_validation_data_structures.md)** - Data structures for validation and alignment testing
- **[Validation Framework Guide](../0_developer_guide/validation_framework_guide.md)** - Comprehensive guide to the validation framework and testing patterns

### **Cross-Reference Integration**

This design document serves as the **architectural foundation** for the search space management improvements detailed in the implementation plans. The design principles established here directly inform:

1. **Implementation Strategy**: The dual search space architecture guides the phased implementation approach in the improvement plan
2. **Violation Remediation**: The specific violations identified here provide the technical requirements for the workspace system compliance fixes
3. **Integration Patterns**: The separation of concerns principles establish the integration patterns used across all related refactoring efforts
4. **Quality Standards**: The redundancy reduction principles align with the code quality standards applied in the config field management refactoring

### **Document Relationships**

```
Search Space Management Design (This Document)
├── Informs → Search Space Management Improvement Plan
├── Integrates → Config Field Management Refactoring Design  
├── References → Code Redundancy Evaluation Guide
├── Validates → Config Field Management System Analysis
└── Implements → Unified Step Catalog System Implementation Plan
```

This comprehensive reference network ensures consistency across all related documentation and provides clear traceability between design decisions, analysis findings, and implementation plans.
