---
tags:
  - design
  - config_discovery
  - simple_solution
  - essential_functionality
keywords:
  - config class auto discovery
  - directory scanning
  - AST parsing
  - build_complete_config_classes
topics:
  - configuration class auto-discovery
  - simple directory scanning
  - essential functionality only
language: python
date of note: 2025-09-16
---

# Configuration Class Auto-Discovery Design

## Executive Summary

This document presents a **simple, focused solution** for automatically discovering configuration classes in two specific directories:
1. `src/cursus/steps/configs` (core configs - always scanned)
2. `development/projects/{project_id}/src/cursus_dev/steps/configs` (workspace configs - if project_id provided)

### Current State Analysis

**Problem**: The `build_complete_config_classes()` function has a TODO for scanning `cursus/steps` directories.

**Solution**: Simple directory scanning with AST-based class detection to automatically discover and register config classes.

### Key Demand Validation

Following the **Code Redundancy Evaluation Guide**:
- ✅ **Real User Need**: TODO exists in production code
- ✅ **Simple Problem**: Just need to scan two specific directory patterns
- ✅ **Avoid Over-Engineering**: No complex workspace management, precedence rules, or theoretical features
- ✅ **Target 15-25% Redundancy**: Minimal, focused implementation

## Simple Solution Design

### Core Implementation

```python
# src/cursus/core/config_fields/config_auto_discovery.py
import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Optional

logger = logging.getLogger(__name__)

def discover_config_classes(project_id: Optional[str] = None) -> Dict[str, Type]:
    """
    Simple config class discovery for two specific paths:
    1. src/cursus/steps/configs (always scanned)
    2. development/projects/{project_id}/src/cursus_dev/steps/configs (if project_id provided)
    
    Args:
        project_id: Optional project ID for workspace configs
        
    Returns:
        Dict mapping class names to class types
    """
    discovered_classes = {}
    
    # Always scan core configs
    core_dir = Path("src/cursus/steps/configs")
    if core_dir.exists():
        try:
            classes = _scan_directory(core_dir)
            discovered_classes.update(classes)
            logger.info(f"Found {len(classes)} config classes in {core_dir}")
        except Exception as e:
            logger.warning(f"Error scanning {core_dir}: {e}")
    
    # Scan workspace configs if project_id provided
    if project_id:
        workspace_dir = Path(f"development/projects/{project_id}/src/cursus_dev/steps/configs")
        if workspace_dir.exists():
            try:
                classes = _scan_directory(workspace_dir)
                discovered_classes.update(classes)  # Workspace configs override core
                logger.info(f"Found {len(classes)} config classes in {workspace_dir}")
            except Exception as e:
                logger.warning(f"Error scanning {workspace_dir}: {e}")
    
    return discovered_classes

def _scan_directory(directory: Path) -> Dict[str, Type]:
    """Scan a directory for config classes."""
    classes = {}
    
    for py_file in directory.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        try:
            file_classes = _extract_config_classes(py_file)
            classes.update(file_classes)
        except Exception as e:
            logger.warning(f"Error processing {py_file}: {e}")
    
    return classes

def _extract_config_classes(file_path: Path) -> Dict[str, Type]:
    """Extract config classes from a Python file using AST."""
    classes = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_config_class(node):
                try:
                    # Import the class
                    module_path = _file_to_module_path(file_path)
                    module = importlib.import_module(module_path)
                    class_type = getattr(module, node.name)
                    classes[node.name] = class_type
                except Exception as e:
                    logger.warning(f"Failed to import {node.name} from {file_path}: {e}")
    
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
    
    return classes

def _is_config_class(class_node: ast.ClassDef) -> bool:
    """Check if a class is a config class."""
    # Check inheritance
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            if base.id in {'BasePipelineConfig', 'ProcessingStepConfigBase', 'BaseModel'}:
                return True
        elif isinstance(base, ast.Attribute):
            if base.attr in {'BasePipelineConfig', 'ProcessingStepConfigBase', 'BaseModel'}:
                return True
    
    # Check naming pattern
    return class_node.name.endswith('Config') or class_node.name.endswith('Configuration')

def _file_to_module_path(file_path: Path) -> str:
    """Convert file path to Python module path."""
    parts = file_path.parts
    
    # Find src directory
    if 'src' in parts:
        src_idx = parts.index('src')
        module_parts = parts[src_idx + 1:]
    else:
        # Fallback
        module_parts = parts[-3:] if len(parts) >= 3 else parts
    
    # Remove .py extension
    if module_parts[-1].endswith('.py'):
        module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
    
    return '.'.join(module_parts)
```

### Enhanced build_complete_config_classes Function

```python
# src/cursus/core/config_fields/config_class_store.py (updated)
def build_complete_config_classes(
    project_id: Optional[str] = None,
    enable_auto_discovery: bool = True
) -> Dict[str, Type]:
    """
    Build complete mapping of config classes from all sources.
    
    Now includes automatic discovery addressing the TODO.
    
    Args:
        project_id: Optional project ID for workspace configs
        enable_auto_discovery: Whether to enable auto-discovery
        
    Returns:
        Dict mapping class names to class types
    """
    # Start with manually registered classes
    config_classes = ConfigClassStore.get_all_classes()
    
    if enable_auto_discovery:
        try:
            from .config_auto_discovery import discover_config_classes
            
            # Discover additional classes
            discovered = discover_config_classes(project_id)
            
            # Add discovered classes (manual registration takes precedence)
            for name, cls in discovered.items():
                if name not in config_classes:
                    config_classes[name] = cls
                    ConfigClassStore.register(cls)
            
            logger.info(f"Auto-discovery found {len(discovered)} config classes")
            
        except Exception as e:
            logger.warning(f"Auto-discovery failed, using manual registration only: {e}")
    
    return config_classes
```

## Usage Examples

### Basic Usage (Core Configs Only)

```python
# Scans only src/cursus/steps/configs
config_classes = build_complete_config_classes()
print(f"Found {len(config_classes)} config classes")
```

### With Project ID

```python
# Scans both core and workspace configs
config_classes = build_complete_config_classes(project_id="project_alpha")
print(f"Found {len(config_classes)} config classes")
```

### Direct Discovery Function

```python
from cursus.core.config_fields.config_auto_discovery import discover_config_classes

# Core only
core_classes = discover_config_classes()

# With workspace
workspace_classes = discover_config_classes("project_alpha")
```

## Architecture Overview

### System Integration Architecture

```mermaid
graph TB
    subgraph "Unified Step Catalog System"
        subgraph "Discovery Layer"
            SC[Step Catalog]
            QE[Query Engine]
        end
        
        subgraph "Foundation Layer"
            URM[Unified Registry Manager]
            CDM[Contract Discovery Manager]
            CCAD[Config Class Auto-Discovery] %% This system
        end
        
        subgraph "Workspace Layer"
            WM[Workspace Manager]
            WD[Workspace Discovery]
        end
    end
    
    subgraph "Configuration System"
        CCS[ConfigClassStore]
        BCCF[build_complete_config_classes]
        TD[Type Detection]
    end
    
    %% Integration connections
    SC -.-> CCAD
    URM --> CCAD
    CCAD --> CCS
    CCAD --> BCCF
    WM --> CCAD
    WD --> CCAD
    
    %% Data flow
    CCAD -->|"Discovered Classes"| CCS
    CCS -->|"Registered Classes"| BCCF
    BCCF -->|"Complete Class Map"| TD
    
    classDef discoveryLayer fill:#e1f5fe
    classDef foundationLayer fill:#e8f5e8
    classDef workspaceLayer fill:#f3e5f5
    classDef configSystem fill:#fff3e0
    
    class SC,QE discoveryLayer
    class URM,CDM,CCAD foundationLayer
    class WM,WD workspaceLayer
    class CCS,BCCF,TD configSystem
```

### Core Components

#### **1. Workspace-Aware Config Discovery**
**Purpose**: Discover config classes across multiple workspaces with configurable scanning
**Responsibility**: File system scanning, AST parsing, and workspace precedence

```python
class WorkspaceAwareConfigDiscovery:
    """Workspace-aware configuration class discovery system."""
    
    def discover_config_classes(
        self,
        additional_scan_dirs: Optional[List[str]] = None,
        workspace_root: Optional[str] = None,
        enable_workspace_discovery: bool = True
    ) -> Dict[str, Type]:
        """Discover config classes with workspace awareness."""
        
    def scan_directory(self, directory: Path) -> Dict[str, Type]:
        """Scan a directory for config classes using AST parsing."""
        
    def resolve_workspace_precedence(
        self, 
        discovered_classes: Dict[str, List[Tuple[Type, str]]]
    ) -> Dict[str, Type]:
        """Resolve conflicts using workspace precedence rules."""
```

#### **2. AST-Based Class Detection**
**Purpose**: Identify config classes without importing modules
**Responsibility**: Safe class detection, inheritance analysis, and metadata extraction

```python
class ConfigClassDetector:
    """AST-based configuration class detection."""
    
    def detect_config_classes(self, file_path: Path) -> List[ConfigClassInfo]:
        """Detect config classes in a Python file using AST parsing."""
        
    def is_config_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class is a configuration class based on inheritance."""
        
    def extract_class_metadata(self, class_node: ast.ClassDef) -> ConfigClassMetadata:
        """Extract metadata from a config class definition."""
```

#### **3. Enhanced build_complete_config_classes**
**Purpose**: Integrate auto-discovery with existing configuration system
**Responsibility**: Coordinate discovery, registration, and caching

```python
def build_complete_config_classes(
    additional_scan_dirs: Optional[List[str]] = None,
    workspace_root: Optional[str] = None,
    enable_auto_discovery: bool = True,
    cache_results: bool = True
) -> Dict[str, Type]:
    """Build complete mapping of config classes with auto-discovery."""
```

## Detailed Design

### Data Models

```python
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List, Dict, Any, Type
from datetime import datetime

class ConfigClassMetadata(BaseModel):
    """Metadata for discovered configuration classes."""
    class_name: str = Field(..., description="Name of the configuration class")
    module_path: str = Field(..., description="Python module path")
    file_path: Path = Field(..., description="File system path")
    workspace_id: str = Field(..., description="Workspace identifier")
    base_classes: List[str] = Field(default_factory=list, description="Base class names")
    docstring: Optional[str] = Field(None, description="Class docstring")
    discovered_at: datetime = Field(default_factory=datetime.now, description="Discovery timestamp")
    
    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True
    }

class ConfigClassInfo(BaseModel):
    """Complete information about a discovered config class."""
    metadata: ConfigClassMetadata = Field(..., description="Class metadata")
    class_type: Optional[Type] = Field(None, description="Actual class type (lazy loaded)")
    import_error: Optional[str] = Field(None, description="Import error if any")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def load_class(self) -> Optional[Type]:
        """Lazy load the actual class type."""
        if self.class_type is not None:
            return self.class_type
            
        if self.import_error is not None:
            return None
            
        try:
            module = importlib.import_module(self.metadata.module_path)
            self.class_type = getattr(module, self.metadata.class_name)
            return self.class_type
        except Exception as e:
            self.import_error = str(e)
            return None

class DiscoveryResult(BaseModel):
    """Result of configuration class discovery operation."""
    discovered_classes: Dict[str, ConfigClassInfo] = Field(default_factory=dict, description="Discovered classes")
    scan_directories: List[Path] = Field(default_factory=list, description="Directories scanned")
    workspace_precedence: List[str] = Field(default_factory=list, description="Workspace precedence order")
    scan_duration: float = Field(..., description="Time taken for discovery in seconds")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during discovery")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
```

### Core Implementation

#### **1. Workspace-Aware Discovery Engine**

```python
# src/cursus/core/config_fields/workspace_auto_discovery.py
import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Tuple, Set
from datetime import datetime
import time

from .config_class_store import ConfigClassStore
from ...workspace.core.config import WorkspaceStepDefinition

logger = logging.getLogger(__name__)

class WorkspaceAwareConfigDiscovery:
    """Workspace-aware configuration class discovery system."""
    
    # Base classes that indicate a config class
    CONFIG_BASE_CLASSES = {
        'BasePipelineConfig',
        'ProcessingStepConfigBase',
        'BaseModel',  # Pydantic base
    }
    
    # Default directories to scan (relative to workspace root)
    DEFAULT_SCAN_PATHS = [
        'src/cursus/steps/configs',
        'src/cursus_dev/steps/configs',  # Developer workspace pattern
        'cursus/steps/configs',  # Alternative structure
    ]
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.detector = ConfigClassDetector()
        self._cache: Dict[str, DiscoveryResult] = {}
        
    def discover_config_classes(
        self,
        additional_scan_dirs: Optional[List[str]] = None,
        workspace_root: Optional[str] = None,
        enable_workspace_discovery: bool = True
    ) -> Dict[str, Type]:
        """
        Discover config classes with workspace awareness.
        
        Args:
            additional_scan_dirs: Additional directories to scan
            workspace_root: Override workspace root
            enable_workspace_discovery: Enable multi-workspace discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        start_time = time.time()
        
        # Determine workspace root
        effective_workspace_root = Path(workspace_root) if workspace_root else self.workspace_root
        
        # Build scan directory list with precedence
        scan_dirs = self._build_scan_directory_list(
            effective_workspace_root,
            additional_scan_dirs,
            enable_workspace_discovery
        )
        
        # Discover classes from all directories
        all_discovered = {}
        errors = []
        
        for scan_dir in scan_dirs:
            try:
                discovered = self._scan_directory_for_configs(scan_dir)
                all_discovered.update(discovered)
            except Exception as e:
                error_msg = f"Error scanning {scan_dir}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # Resolve workspace precedence conflicts
        resolved_classes = self._resolve_workspace_precedence(all_discovered)
        
        # Create discovery result
        result = DiscoveryResult(
            discovered_classes={name: ConfigClassInfo(metadata=info.metadata, class_type=info.class_type) 
                              for name, info in resolved_classes.items()},
            scan_directories=scan_dirs,
            scan_duration=time.time() - start_time,
            errors=errors
        )
        
        # Cache result
        cache_key = self._generate_cache_key(effective_workspace_root, additional_scan_dirs)
        self._cache[cache_key] = result
        
        # Return class types only
        return {name: info.class_type for name, info in resolved_classes.items() 
                if info.class_type is not None}
    
    def _build_scan_directory_list(
        self,
        workspace_root: Path,
        additional_scan_dirs: Optional[List[str]],
        enable_workspace_discovery: bool
    ) -> List[Path]:
        """Build prioritized list of directories to scan."""
        scan_dirs = []
        
        # 1. Additional user-specified directories (highest priority)
        if additional_scan_dirs:
            for dir_path in additional_scan_dirs:
                path = Path(dir_path)
                if not path.is_absolute():
                    path = workspace_root / path
                if path.exists() and path.is_dir():
                    scan_dirs.append(path)
        
        # 2. Workspace-specific directories (if enabled)
        if enable_workspace_discovery:
            workspace_dirs = self._discover_workspace_config_dirs(workspace_root)
            scan_dirs.extend(workspace_dirs)
        
        # 3. Default directories (lowest priority)
        for default_path in self.DEFAULT_SCAN_PATHS:
            path = workspace_root / default_path
            if path.exists() and path.is_dir():
                scan_dirs.append(path)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dirs = []
        for dir_path in scan_dirs:
            if dir_path not in seen:
                seen.add(dir_path)
                unique_dirs.append(dir_path)
        
        return unique_dirs
    
    def _discover_workspace_config_dirs(self, workspace_root: Path) -> List[Path]:
        """Discover configuration directories in workspace structure."""
        workspace_dirs = []
        
        # Look for developer workspaces
        dev_workspace_pattern = workspace_root / "workspaces" / "*" / "src" / "cursus_dev" / "steps" / "configs"
        for workspace_dir in workspace_root.glob("workspaces/*/src/cursus_dev/steps/configs"):
            if workspace_dir.is_dir():
                workspace_dirs.append(workspace_dir)
        
        # Look for shared workspace
        shared_workspace = workspace_root / "shared" / "src" / "cursus" / "steps" / "configs"
        if shared_workspace.exists() and shared_workspace.is_dir():
            workspace_dirs.append(shared_workspace)
        
        return workspace_dirs
    
    def _scan_directory_for_configs(self, directory: Path) -> Dict[str, ConfigClassInfo]:
        """Scan a directory for configuration classes."""
        discovered = {}
        
        # Get workspace ID from directory path
        workspace_id = self._extract_workspace_id(directory)
        
        # Scan Python files
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                config_classes = self.detector.detect_config_classes(py_file)
                
                for class_info in config_classes:
                    # Update workspace ID
                    class_info.metadata.workspace_id = workspace_id
                    
                    # Store with potential for conflict resolution
                    class_name = class_info.metadata.class_name
                    if class_name in discovered:
                        # Handle conflict - later we'll resolve by workspace precedence
                        logger.debug(f"Duplicate config class {class_name} found in {py_file}")
                    
                    discovered[class_name] = class_info
                    
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")
                continue
        
        return discovered
    
    def _resolve_workspace_precedence(
        self, 
        discovered_classes: Dict[str, ConfigClassInfo]
    ) -> Dict[str, ConfigClassInfo]:
        """Resolve conflicts using workspace precedence rules."""
        # For now, simple last-wins strategy
        # In the future, could implement sophisticated precedence rules
        return discovered_classes
    
    def _extract_workspace_id(self, directory: Path) -> str:
        """Extract workspace ID from directory path."""
        path_parts = directory.parts
        
        # Look for workspace patterns
        if "workspaces" in path_parts:
            workspace_idx = path_parts.index("workspaces")
            if workspace_idx + 1 < len(path_parts):
                return f"developer_{path_parts[workspace_idx + 1]}"
        
        if "shared" in path_parts:
            return "shared"
        
        # Default to core workspace
        return "core"
    
    def _generate_cache_key(
        self, 
        workspace_root: Path, 
        additional_scan_dirs: Optional[List[str]]
    ) -> str:
        """Generate cache key for discovery results."""
        key_parts = [str(workspace_root)]
        if additional_scan_dirs:
            key_parts.extend(sorted(additional_scan_dirs))
        return "|".join(key_parts)

class ConfigClassDetector:
    """AST-based configuration class detection."""
    
    def detect_config_classes(self, file_path: Path) -> List[ConfigClassInfo]:
        """Detect config classes in a Python file using AST parsing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(file_path))
            config_classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if self.is_config_class(node):
                        metadata = self.extract_class_metadata(node, file_path)
                        config_info = ConfigClassInfo(metadata=metadata)
                        config_classes.append(config_info)
            
            return config_classes
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return []
    
    def is_config_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class is a configuration class based on inheritance."""
        # Check base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id in WorkspaceAwareConfigDiscovery.CONFIG_BASE_CLASSES:
                    return True
            elif isinstance(base, ast.Attribute):
                # Handle qualified names like module.BaseClass
                if base.attr in WorkspaceAwareConfigDiscovery.CONFIG_BASE_CLASSES:
                    return True
        
        # Check class name patterns
        class_name = class_node.name
        if class_name.endswith('Config') or class_name.endswith('Configuration'):
            return True
        
        return False
    
    def extract_class_metadata(self, class_node: ast.ClassDef, file_path: Path) -> ConfigClassMetadata:
        """Extract metadata from a config class definition."""
        # Extract module path from file path
        module_path = self._file_path_to_module_path(file_path)
        
        # Extract base class names
        base_classes = []
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(base.attr)
        
        # Extract docstring
        docstring = None
        if (class_node.body and 
            isinstance(class_node.body[0], ast.Expr) and 
            isinstance(class_node.body[0].value, ast.Constant) and 
            isinstance(class_node.body[0].value.value, str)):
            docstring = class_node.body[0].value.value
        
        return ConfigClassMetadata(
            class_name=class_node.name,
            module_path=module_path,
            file_path=file_path,
            workspace_id="",  # Will be set by caller
            base_classes=base_classes,
            docstring=docstring
        )
    
    def _file_path_to_module_path(self, file_path: Path) -> str:
        """Convert file path to Python module path."""
        # Find the src directory or cursus directory
        parts = file_path.parts
        
        # Look for src directory
        if 'src' in parts:
            src_idx = parts.index('src')
            module_parts = parts[src_idx + 1:]
        else:
            # Fallback to cursus directory
            if 'cursus' in parts:
                cursus_idx = parts.index('cursus')
                module_parts = parts[cursus_idx:]
            else:
                # Use the last few parts as fallback
                module_parts = parts[-3:] if len(parts) >= 3 else parts
        
        # Remove .py extension
        if module_parts[-1].endswith('.py'):
            module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
        
        return '.'.join(module_parts)
```

#### **2. Enhanced build_complete_config_classes Function**

```python
# src/cursus/core/config_fields/config_class_store.py (updated)
def build_complete_config_classes(
    additional_scan_dirs: Optional[List[str]] = None,
    workspace_root: Optional[str] = None,
    enable_auto_discovery: bool = True,
    cache_results: bool = True
) -> Dict[str, Type]:
    """
    Build a complete mapping of config classes from all available sources.
    
    This function now includes automatic discovery of config classes across
    workspaces, addressing the TODO for scanning cursus/steps directories.
    
    Args:
        additional_scan_dirs: Additional directories to scan for config classes
        workspace_root: Root directory for workspace-aware discovery
        enable_auto_discovery: Whether to enable automatic discovery
        cache_results: Whether to cache discovery results
        
    Returns:
        dict: Mapping of class names to class objects
    """
    # Start with manually registered classes (highest priority)
    config_classes = ConfigClassStore.get_all_classes()
    
    if enable_auto_discovery:
        try:
            # Import here to avoid circular imports
            from .workspace_auto_discovery import WorkspaceAwareConfigDiscovery
            
            # Create discovery engine
            discovery_engine = WorkspaceAwareConfigDiscovery(
                workspace_root=Path(workspace_root) if workspace_root else None
            )
            
            # Discover additional classes
            discovered_classes = discovery_engine.discover_config_classes(
                additional_scan_dirs=additional_scan_dirs,
                workspace_root=workspace_root,
                enable_workspace_discovery=True
            )
            
            # Merge discovered classes (manual registration takes precedence)
            for class_name, class_type in discovered_classes.items():
                if class_name not in config_classes:
                    config_classes[class_name] = class_type
                    # Also register in the store for consistency
                    ConfigClassStore.register(class_type)
            
            logger.info(f"Auto-discovery found {len(discovered_classes)} config classes")
            
        except Exception as e:
            logger.warning(f"Auto-discovery failed, using manual registration only: {e}")
    
    return config_classes
```

#### **3. Integration with ConfigClassDetector**

```python
# src/cursus/core/config_fields/config_class_detector.py (updated)
def detect_config_classes_from_json(config_path: str) -> Dict[str, Type]:
    """
    Detect required config classes from a configuration JSON file.
    Now uses enhanced build_complete_config_classes with auto-discovery.
    """
    logger = logging.getLogger(__name__)

    try:
        # Verify the file exists
        if not Path(config_path).is_file():
            logger.error(f"Configuration file not found: {config_path}")
            return build_complete_config_classes()

        # Read and parse the JSON file
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Extract required class names
        required_class_names = ConfigClassDetector._extract_class_names(
            config_data, logger
        )

        if not required_class_names:
            logger.warning("No config class names found in configuration file")
            # Fallback to loading all classes with auto-discovery
            return build_complete_config_classes()

        logger.info(
            f"Detected {len(required_class_names)} required config classes in configuration file"
        )

        # Get all available classes using enhanced discovery
        all_available_classes = build_complete_config_classes()
        required_classes = {}

        # Only keep classes that are actually used in the config file
        for class_name, class_type in all_available_classes.items():
            if class_name in required_class_names:
                required_classes[class_name] = class_type

        # Always include essential base classes
        for essential_class in ConfigClassDetector.ESSENTIAL_CLASSES:
            if (
                essential_class not in required_classes
                and essential_class in all_available_classes
            ):
                required_classes[essential_class] = all_available_classes[
                    essential_class
                ]

        # Report on any missing classes that couldn't be loaded
        missing_classes = required_class_names - set(required_classes.keys())
        if missing_classes:
            logger.warning(
                f"Could not load {len(missing_classes)} required classes: {missing_classes}"
            )

        logger.info(
            f"Successfully loaded {len(required_classes)} of {len(required_class_names)} required classes"
        )

        return required_classes

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing configuration file: {e}")
        logger.warning("Falling back to loading all available config classes with auto-discovery")
        # Fallback to loading all classes with auto-discovery
        return build_complete_config_classes()
```

## Usage Examples

### Basic Usage

```python
# Default behavior - scans cursus/steps automatically
config_classes = build_complete_config_classes()
print(f"Found {len(config_classes)} config classes")
```

### Workspace-Aware Usage

```python
# Workspace-aware discovery with additional directories
config_classes = build_complete_config_classes(
    workspace_root="/path/to/workspace",
    additional_scan_dirs=[
        "/custom/configs",
        "shared/configurations"
    ]
)
```

### Developer Workspace Usage

```python
# From WorkspaceStepDefinition
step_def = WorkspaceStepDefinition(...)
config_classes = build_complete_config_classes(
    workspace_root=step_def.workspace_root,
    additional_scan_dirs=[step_def.get_workspace_path("custom_configs")]
)
```

### Environment Variable Configuration

```python
import os

# Configure via environment variables
os.environ['CURSUS_CONFIG_SCAN_DIRS'] = '/custom/configs:/shared/configs'
os.environ['CURSUS_WORKSPACE_ROOT'] = '/path/to/workspace'

# Auto-discovery will use environment configuration
config_classes = build_complete_config_classes()
```

## Performance Considerations

### Optimization Strategies

#### **1. AST-Based Pre-filtering**
- Use AST parsing to identify config classes before importing
- Avoid expensive module imports for non-config files
- Cache AST parsing results for unchanged files

#### **2. Lazy Loading**
- Load class metadata immediately, defer actual class import
- Import classes only when actually needed
- Use weak references to avoid memory leaks

#### **3. Intelligent Caching**
- Cache discovery results with file modification time tracking
- Invalidate cache entries when source files change
- Use workspace-specific cache keys

#### **4. Incremental Discovery**
- Support incremental updates for changed files
- Avoid full workspace scans when possible
- Track file dependencies for efficient invalidation

### Performance Targets

- **Discovery Time**: <5 seconds for 1000+ config files
- **Memory Usage**: <50MB for typical workspace
- **Cache Hit Rate**: >90% for repeated discoveries
- **Incremental Update**: <1 second for single file changes

## Error Handling and Resilience

### Error Recovery Strategies

#### **1. Graceful Degradation**
```python
class ConfigDiscoveryErrorHandler:
    """Handles discovery errors with graceful fallbacks."""
    
    def handle_import_error(self, class_info: ConfigClassInfo) -> Optional[Type]:
        """Handle class import errors with fallback strategies."""
        try:
            # Try alternative import paths
            return self._try_alternative_imports(class_info)
        except Exception:
            # Log error and continue with other classes
            logger.warning(f"Failed to import {class_info.metadata.class_name}")
            return None
    
    def handle_workspace_unavailable(self, workspace_path: Path) -> List[Path]:
        """Handle unavailable workspace with cached data."""
        cached_paths = self._get_cached_workspace_paths(workspace_path)
        if cached_paths:
            logger.info(f"Using cached paths for unavailable workspace {workspace_path}")
            return cached_paths
        return []
```

#### **2. Partial Discovery Support**
- Continue discovery even if some directories fail
- Report errors but don't fail entire operation
- Provide detailed error information for debugging

#### **3. Fallback Mechanisms**
- Fall back to manual registration if auto-discovery fails
- Use cached results if file system is unavailable
- Provide minimal functionality even in error conditions

## Integration Points

### Unified Step Catalog Integration

The config class auto-discovery system integrates seamlessly with the Unified Step Catalog System:

#### **1. Foundation Layer Integration**
```python
# Integration with registry system
class UnifiedRegistryManager:
    def __init__(self):
        self.config_discovery = WorkspaceAwareConfigDiscovery()
    
    def get_step_config_class(self, step_name: str) -> Optional[Type]:
        """Get config class for a step using auto-discovery."""
        config_classes = self.config_discovery.discover_config_classes()
        
        # Try direct match first
        if step_name in config_classes:
            return config_classes[step_name]
        
        # Try pattern matching
        for class_name, class_type in config_classes.items():
            if self._matches_step_pattern(step_name, class_name):
                return class_type
        
        return None
```

#### **2. Workspace-Aware Integration**
```python
# Integration with workspace management
class WorkspaceAwareCatalogManager:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.config_discovery = WorkspaceAwareConfigDiscovery(workspace_root)
    
    def get_workspace_config_classes(self, workspace_id: str) -> Dict[str, Type]:
        """Get config classes specific to a workspace."""
        all_classes = self.config_discovery.discover_config_classes()
        
        # Filter by workspace
        workspace_classes = {}
        for class_name, class_type in all_classes.items():
            if self._belongs_to_workspace(class_type, workspace_id):
                workspace_classes[class_name] = class_type
        
        return workspace_classes
```

### Configuration System Integration

#### **1. Type-Aware Serialization**
```python
# Enhanced type-aware serialization with auto-discovery
class TypeAwareConfigSerializer:
    def __init__(self):
        self.config_classes = build_complete_config_classes()
    
    def serialize_with_discovery(self, config_data: Dict[str, Any]) -> str:
        """Serialize config with automatic class discovery."""
        # Auto-discover required classes
        required_classes = self._detect_required_classes(config_data)
        
        # Ensure all required classes are available
        for class_name in required_classes:
            if class_name not in self.config_classes:
                # Trigger re-discovery
                self.config_classes = build_complete_config_classes()
                break
        
        return self._serialize_with_types(config_data, self.config_classes)
```

#### **2. Configuration Validation**
```python
# Enhanced validation with auto-discovery
class ConfigValidator:
    def validate_with_discovery(self, config_path: str) -> ValidationResult:
        """Validate configuration with automatic class discovery."""
        # Use enhanced detection with auto-discovery
        config_classes = detect_config_classes_from_json(config_path)
        
        # Validate completeness
        validation_result = ValidationResult()
        for class_name, class_type in config_classes.items():
            if not self._validate_class_completeness(class_type):
                validation_result.add_error(f"Incomplete config class: {class_name}")
        
        return validation_result
```

## Testing Strategy

### Unit Testing

#### **1. Discovery Engine Tests**
```python
class TestWorkspaceAwareConfigDiscovery:
    """Test workspace-aware config discovery."""
    
    def test_basic_discovery(self):
        """Test basic config class discovery."""
        discovery = WorkspaceAwareConfigDiscovery(test_workspace_root)
        classes = discovery.discover_config_classes()
        
        assert "XGBoostTrainingConfig" in classes
        assert "ProcessingStepConfigBase" in classes
    
    def test_workspace_precedence(self):
        """Test workspace precedence resolution."""
        discovery = WorkspaceAwareConfigDiscovery(multi_workspace_root)
        classes = discovery.discover_config_classes()
        
        # Developer workspace should take precedence
        config_class = classes["SharedConfig"]
        assert config_class.__module__.startswith("cursus_dev")
    
    def test_additional_scan_dirs(self):
        """Test additional scan directories."""
        discovery = WorkspaceAwareConfigDiscovery(test_workspace_root)
        classes = discovery.discover_config_classes(
            additional_scan_dirs=["/custom/configs"]
        )
        
        assert "CustomConfig" in classes
```

#### **2. AST Detection Tests**
```python
class TestConfigClassDetector:
    """Test AST-based config class detection."""
    
    def test_inheritance_detection(self):
        """Test detection based on inheritance."""
        detector = ConfigClassDetector()
        
        # Create test file with config class
        test_code = '''
class TestConfig(BasePipelineConfig):
    """Test configuration class."""
    field1: str = "value"
'''
        
        with temp_file(test_code) as file_path:
            classes = detector.detect_config_classes(file_path)
            assert len(classes) == 1
            assert classes[0].metadata.class_name == "TestConfig"
    
    def test_name_pattern_detection(self):
        """Test detection based on naming patterns."""
        detector = ConfigClassDetector()
        
        test_code = '''
class MyCustomConfig:
    """Custom configuration class."""
    pass
'''
        
        with temp_file(test_code) as file_path:
            classes = detector.detect_config_classes(file_path)
            assert len(classes) == 1
```

### Integration Testing

#### **1. End-to-End Discovery Tests**
```python
class TestConfigDiscoveryIntegration:
    """Test complete config discovery integration."""
    
    def test_build_complete_config_classes(self):
        """Test enhanced build_complete_config_classes function."""
        classes = build_complete_config_classes(
            workspace_root=test_workspace_root,
            enable_auto_discovery=True
        )
        
        # Should include manually registered classes
        assert "BasePipelineConfig" in classes
        
        # Should include auto-discovered classes
        assert "XGBoostTrainingConfig" in classes
        assert "ProcessingStepConfigBase" in classes
    
    def test_json_detection_integration(self):
        """Test integration with JSON config detection."""
        classes = detect_config_classes_from_json(test_config_path)
        
        # Should auto-discover required classes
        assert len(classes) > 0
        assert all(isinstance(cls, type) for cls in classes.values())
```

### Performance Testing

#### **1. Discovery Performance Tests**
```python
class TestConfigDiscoveryPerformance:
    """Test config discovery performance."""
    
    def test_large_workspace_discovery(self):
        """Test discovery performance with large workspace."""
        discovery = WorkspaceAwareConfigDiscovery(large_workspace_root)
        
        start_time = time.time()
        classes = discovery.discover_config_classes()
        discovery_time = time.time() - start_time
        
        assert discovery_time < 5.0  # <5 seconds requirement
        assert len(classes) > 100  # Should find many classes
    
    def test_cache_effectiveness(self):
        """Test caching effectiveness."""
        discovery = WorkspaceAwareConfigDiscovery(test_workspace_root)
        
        # First discovery
        start_time = time.time()
        classes1 = discovery.discover_config_classes()
        first_time = time.time() - start_time
        
        # Second discovery (should use cache)
        start_time = time.time()
        classes2 = discovery.discover_config_classes()
        second_time = time.time() - start_time
        
        assert second_time < first_time * 0.1  # >90% improvement
        assert classes1 == classes2  # Same results
```

## Migration Strategy

### Phased Implementation

#### **Phase 1: Core Auto-Discovery (1 week)**
1. Implement `WorkspaceAwareConfigDiscovery` class
2. Implement `ConfigClassDetector` with AST parsing
3. Create basic data models and error handling
4. Add unit tests for core functionality

#### **Phase 2: Integration (1 week)**
1. Update `build_complete_config_classes()` function
2. Integrate with existing `ConfigClassDetector`
3. Add workspace-aware directory discovery
4. Implement caching and performance optimizations

#### **Phase 3: Advanced Features (1 week)**
1. Add environment variable configuration
2. Implement workspace precedence resolution
3. Add comprehensive error handling and fallbacks
4. Create integration tests and performance benchmarks

#### **Phase 4: Documentation and Deployment (0.5 weeks)**
1. Update documentation and examples
2. Create migration guide for existing users
3. Deploy with feature flags for gradual rollout

### Backward Compatibility

The system maintains 100% backward compatibility:

- **Existing decorators continue to work**: `@ConfigClassStore.register` still functions
- **Manual registration takes precedence**: Manually registered classes override auto-discovered ones
- **Graceful fallback**: If auto-discovery fails, system falls back to manual registration
- **Optional feature**: Auto-discovery can be disabled via parameter

## Monitoring and Observability

### Metrics Collection

```python
class ConfigDiscoveryMetrics:
    """Collect metrics for config discovery operations."""
    
    def __init__(self):
        self.discovery_count = 0
        self.discovery_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0
    
    def record_discovery(self, duration: float, classes_found: int, from_cache: bool):
        """Record discovery operation metrics."""
        self.discovery_count += 1
        self.discovery_times.append(duration)
        
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "total_discoveries": self.discovery_count,
            "average_discovery_time": sum(self.discovery_times) / len(self.discovery_times),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses),
            "error_rate": self.error_count / self.discovery_count
        }
```

### Health Checks

```python
class ConfigDiscoveryHealthCheck:
    """Health check for config discovery system."""
    
    def check_discovery_health(self) -> HealthStatus:
        """Check if discovery system is healthy."""
        try:
            # Test basic discovery functionality
            discovery = WorkspaceAwareConfigDiscovery()
            test_classes = discovery.discover_config_classes()
            
            if len(test_classes) == 0:
                return HealthStatus.UNHEALTHY("No config classes discovered")
            
            # Test class loading
            for class_name, class_type in test_classes.items():
                if class_type is None:
                    return HealthStatus.DEGRADED(f"Failed to load class: {class_name}")
            
            return HealthStatus.HEALTHY("Config discovery system operational")
            
        except Exception as e:
            return HealthStatus.UNHEALTHY(f"Discovery system error: {e}")
```

## Conclusion

The Configuration Class Auto-Discovery System addresses the critical need for automatic configuration class management in the Cursus ecosystem. By integrating seamlessly with the **Unified Step Catalog System**, it provides:

### **Key Benefits**

#### **1. Reduced Maintenance Burden**
- **Eliminates manual registration**: No more updating `__init__.py` files for new config classes
- **Automatic discovery**: New config classes are automatically found and registered
- **Workspace-aware**: Supports multi-developer environments with workspace precedence

#### **2. Enhanced Developer Experience**
- **Zero configuration**: Works out of the box with sensible defaults
- **Flexible configuration**: Supports additional scan directories and workspace overrides
- **Intelligent caching**: Fast repeated discoveries with cache invalidation
- **Comprehensive error handling**: Graceful degradation with detailed error reporting

#### **3. Seamless Integration**
- **Unified Step Catalog**: Integrates as a foundation layer component
- **Backward compatibility**: Existing decorator-based registration continues to work
- **Configuration system**: Enhances type detection and serialization capabilities
- **Workspace system**: Leverages workspace management for multi-developer support

### **Technical Achievements**

#### **1. Intelligent Discovery**
- **AST-based detection**: Safe class identification without module imports
- **Pattern matching**: Detects config classes by inheritance and naming patterns
- **Metadata extraction**: Rich metadata including docstrings and base classes
- **Lazy loading**: Efficient memory usage with on-demand class loading

#### **2. Workspace-Aware Architecture**
- **Multi-workspace support**: Discovers classes across developer and shared workspaces
- **Precedence resolution**: Handles conflicts with configurable precedence rules
- **Directory scanning**: Flexible directory patterns with user overrides
- **Environment integration**: Supports environment variable configuration

#### **3. Performance Optimization**
- **Caching strategy**: Intelligent caching with file modification tracking
- **Incremental updates**: Efficient updates for changed files
- **Performance targets**: <5 seconds for 1000+ files, <50MB memory usage
- **Scalable architecture**: Designed for growing configuration catalogs

### **Integration with Broader Architecture**

The system serves as a **specialized component** within the Unified Step Catalog System:

- **Foundation Layer**: Provides configuration class discovery for registry operations
- **Discovery Layer**: Supplies configuration metadata for step component queries
- **Workspace Layer**: Integrates with workspace management for multi-developer support

This integration ensures that configuration class discovery is part of a **coherent, scalable component ecosystem** that enables developers to efficiently manage and discover all types of step components.

### **Future Extensibility**

The design provides a solid foundation for future enhancements:

- **Advanced precedence rules**: Sophisticated conflict resolution strategies
- **Semantic analysis**: Enhanced AST parsing for deeper class understanding
- **Integration hooks**: Extension points for custom discovery logic
- **Monitoring integration**: Built-in metrics and health checking capabilities

### **Implementation Readiness**

The design is ready for implementation with:

- **Clear architecture**: Well-defined components with specific responsibilities
- **Comprehensive testing**: Unit, integration, and performance testing strategies
- **Migration strategy**: Phased implementation with backward compatibility
- **Quality assurance**: Error handling, monitoring, and health checking

This Configuration Class Auto-Discovery System transforms the current **manual configuration management** into an **intelligent, automated discovery ecosystem** that scales with the growing Cursus catalog while maintaining simplicity and reliability.

## References

### **Primary Design Documents**

**Unified Step Catalog Integration**:
- **[Unified Step Catalog System Design](./unified_step_catalog_system_design.md)** - Comprehensive design for the broader step catalog system
- **[Unified Step Catalog System Implementation Plan](../2_project_planning/2025-09-10_unified_step_catalog_system_implementation_plan.md)** - Implementation plan for the unified catalog system

**Configuration System References**:
- **[Config](./config.md)** - Core configuration system design
- **[Config Driven Design](./config_driven_design.md)** - Configuration-driven development principles
- **[Config Field Manager Refactoring](./config_field_manager_refactoring.md)** - Configuration field management patterns

### **Workspace-Aware System Integration**

**Core Workspace System**:
- **[Workspace Aware System Master Design](./workspace_aware_system_master_design.md)** - Comprehensive workspace-aware system architecture
- **[Workspace Aware Core System Design](./workspace_aware_core_system_design.md)** - Core workspace management components
- **[Workspace Aware Multi Developer Management Design](./workspace_aware_multi_developer_management_design.md)** - Multi-developer workspace coordination

**Workspace Configuration**:
- **[Workspace Aware Config Manager Design](./workspace_aware_config_manager_design.md)** - Workspace-specific configuration management

### **Discovery and Resolution Patterns**

**Component Discovery**:
- **[Contract Discovery Manager Design](./contract_discovery_manager_design.md)** - Contract discovery mechanisms and patterns
- **[Flexible File Resolver Design](./flexible_file_resolver_design.md)** - Dynamic file discovery and resolution patterns
- **[Dependency Resolution System](./dependency_resolution_system.md)** - Component dependency resolution architecture

### **Registry System Integration**

**Core Registry Design**:
- **[Registry Manager](./registry_manager.md)** - Core registry management system
- **[Registry Single Source of Truth](./registry_single_source_of_truth.md)** - Centralized registry principles
- **[Workspace Aware Distributed Registry Design](./workspace_aware_distributed_registry_design.md)** - Distributed registry across workspaces

### **Quality and Standards**

**Code Quality Framework**:
- **[Code Redundancy Evaluation Guide](./code_redundancy_evaluation_guide.md)** - Framework for assessing architectural efficiency
- **[Design Principles](./design_principles.md)** - Foundational design principles and architectural philosophy
- **[Documentation YAML Frontmatter Standard](./documentation_yaml_frontmatter_standard.md)** - Documentation standards used in this design

### **Architecture Patterns**

**Proven Patterns**:
- **AST-Based Discovery Pattern** - Safe class detection without module imports
- **Workspace Precedence Pattern** - Multi-workspace conflict resolution
- **Lazy Loading Pattern** - Efficient resource utilization
- **Intelligent Caching Pattern** - Performance optimization with invalidation
- **Graceful Degradation Pattern** - Error handling with fallback strategies

### **Related Implementation References**

**Existing Configuration Components**:
- **ConfigClassStore** - Current configuration class registry (`src/cursus/core/config_fields/config_class_store.py`)
- **ConfigClassDetector** - Current configuration class detection (`src/cursus/core/config_fields/config_class_detector.py`)
- **build_complete_config_classes** - Function to be enhanced with auto-discovery

**Workspace System Components**:
- **WorkspaceStepDefinition** - Workspace step definitions (`src/cursus/workspace/core/config.py`)
- **WorkspaceManager** - Workspace management system
- **WorkspaceDiscoveryManager** - Workspace component discovery

This design document provides the foundation for implementing intelligent configuration class discovery that seamlessly integrates with the broader Cursus ecosystem while maintaining simplicity, performance, and reliability.
