---
tags:
  - design
  - architecture
  - path_resolution
  - deployment_portability
  - package_aware
keywords:
  - deployment context agnostic
  - package relative paths
  - path resolution
  - portable configuration
  - runtime context detection
  - serverless deployment
  - container portability
topics:
  - path resolution architecture
  - deployment portability
  - package-aware path management
  - configuration portability
language: python
date of note: 2025-09-22
---

# Deployment-Context-Agnostic Path Resolution Design

## Executive Summary

This document presents a comprehensive design for a deployment-context-agnostic path resolution system that eliminates the critical portability flaw in configuration management across different deployment environments. The current system fails when configurations created in development environments are executed in serverless, container, or package-installed contexts due to working-directory-dependent path resolution.

The proposed solution implements **package-aware path resolution** that resolves paths relative to the package installation location rather than the current working directory, ensuring universal portability across all deployment contexts.

## Problem Statement

### Current System Limitations

The existing path resolution system in the cursus framework suffers from a fundamental architectural flaw: **context-dependent path resolution** that assumes consistent working directories and file system structures across development and deployment environments.

#### Critical Failure Pattern

1. **Development Context**: Configurations created with paths resolved relative to development working directory
2. **Package Distribution**: Configurations packaged with development-context relative paths
3. **Deployment Context**: Same relative paths applied in completely different execution environment
4. **Resolution Failure**: Development-relative paths don't resolve to actual files in deployment environment

#### Specific Context Mismatches

| Context Element | Development Environment | Serverless Environment | Container Environment |
|----------------|------------------------|----------------------|---------------------|
| **Working Directory** | `/workspace/project/` | `/var/task/` | `/app/` |
| **Package Location** | `/workspace/project/src/package/` | `/tmp/package/` | `/usr/local/lib/python3.x/site-packages/package/` |
| **Path Resolution Base** | Development workspace | Lambda runtime directory | Container working directory |
| **File System Structure** | Source directory layout | Installed package layout | Containerized package layout |

### Root Cause Analysis

The core issue lies in the `_convert_to_relative_path()` method that uses `Path.cwd()` (current working directory) as the reference point for path conversion:

```python
# PROBLEMATIC: Context-dependent path resolution
def _convert_to_relative_path(self, path: str) -> str:
    runtime_location = Path.cwd()  # Different in each deployment context
    relative_path = abs_path.relative_to(runtime_location)
    return str(relative_path)
```

This approach creates paths that are only valid in the specific context where they were created, making them fundamentally **non-portable**.

## Design Objectives

### Primary Objectives

1. **Universal Deployment Portability**: Same configuration files work across all deployment contexts
2. **Package-Aware Path Resolution**: Paths resolved relative to package installation location
3. **Runtime Context Independence**: Path resolution independent of current working directory
4. **Zero Breaking Changes**: Complete backward compatibility with existing configurations
5. **Transparent Operation**: No changes required to existing user workflows

### Secondary Objectives

1. **Robust Fallback Mechanisms**: Automatic fallback to absolute paths when package-relative resolution fails
2. **Runtime Context Detection**: Intelligent detection of deployment environment characteristics
3. **Enhanced Debugging**: Comprehensive logging and validation for path resolution troubleshooting
4. **Future Extensibility**: Architecture supports advanced deployment-specific optimizations

## Solution Architecture

### Core Design Principle: Minimal Viable Solution

Based on the code redundancy evaluation guide principles, this design focuses on solving the **core problem** with the **simplest possible solution** that avoids over-engineering while maintaining robustness.

**Key Insight**: The problem is fundamentally simple - we need to find the package root and resolve paths relative to it, not the current working directory.

### Single Component Solution

Instead of multiple complex components, we implement **one focused utility** that solves the core problem:

```python
def get_package_relative_path(absolute_path: str) -> str:
    """
    Convert absolute path to package-relative path for deployment portability.
    
    This is the ONLY function needed to solve the core problem.
    """
    if not absolute_path or not Path(absolute_path).is_absolute():
        return absolute_path  # Already relative or empty
    
    try:
        abs_path = Path(absolute_path)
        
        # Strategy 1: Find package name in path components
        path_parts = abs_path.parts
        
        # Look for common package indicators in path
        package_indicators = ['cursus', 'buyer_abuse_mods_template', 'src']
        
        for indicator in package_indicators:
            if indicator in path_parts:
                indicator_index = path_parts.index(indicator)
                
                # If indicator is 'src', skip it and use next part as package
                if indicator == 'src' and indicator_index + 1 < len(path_parts):
                    package_index = indicator_index + 1
                    relative_parts = path_parts[package_index + 1:]
                else:
                    # Use parts after the package indicator
                    relative_parts = path_parts[indicator_index + 1:]
                
                if relative_parts:
                    return str(Path(*relative_parts))
        
        # Strategy 2: Fallback - return original path
        logger.debug(f"Could not convert to package-relative: {absolute_path}")
        return absolute_path
        
    except Exception as e:
        logger.debug(f"Path conversion failed for {absolute_path}: {e}")
        return absolute_path


def resolve_package_relative_path(relative_path: str) -> str:
    """
    Resolve package-relative path to absolute path in current deployment context.
    
    This is the ONLY other function needed.
    """
    if not relative_path or Path(relative_path).is_absolute():
        return relative_path  # Already absolute or empty
    
    try:
        # Find package root using simple module inspection
        import cursus
        if hasattr(cursus, '__file__') and cursus.__file__:
            package_root = Path(cursus.__file__).parent
            resolved_path = package_root / relative_path
            
            if resolved_path.exists():
                return str(resolved_path.resolve())
        
        # Fallback: return original path
        logger.debug(f"Could not resolve package-relative path: {relative_path}")
        return relative_path
        
    except Exception as e:
        logger.debug(f"Path resolution failed for {relative_path}: {e}")
        return relative_path
```

### Enhanced Configuration Base Class

**Single, focused change** to the existing `BasePipelineConfig`:

```python
class BasePipelineConfig(BaseModel, ABC):
    """Base configuration with deployment-agnostic path resolution."""
    
    # ... existing fields unchanged ...
    
    def _convert_to_relative_path(self, path: str) -> str:
        """
        Convert absolute path to package-relative path (REPLACEMENT METHOD).
        
        This replaces the problematic Path.cwd() approach with package-aware resolution.
        """
        return get_package_relative_path(path)
    
    def get_resolved_path(self, relative_path: str) -> str:
        """
        Resolve package-relative path to absolute path (NEW HELPER METHOD).
        
        Step builders can use this to get absolute paths when needed.
        """
        return resolve_package_relative_path(relative_path)
```

### Why This Simple Solution Works

#### ✅ **Solves the Core Problem**
- Eliminates `Path.cwd()` dependency that causes context mismatch
- Uses package-relative paths that work across all deployment contexts
- Maintains backward compatibility with existing configurations

#### ✅ **Avoids Over-Engineering**
- **No complex class hierarchies** - just two utility functions
- **No deployment context detection** - not needed for the core problem
- **No multiple resolution strategies** - simple approach works for all cases
- **No configuration management** - uses existing configuration system

#### ✅ **Follows Redundancy Guide Principles**
- **Validates Demand**: Solves the actual, validated problem (Lambda deployment failures)
- **Simplest Solution**: Minimal code change that addresses root cause
- **Performance First**: No performance degradation, actually improves performance
- **Maintainable**: Easy to understand, test, and modify

#### ✅ **Robust Fallback Strategy**
- **Level 1**: Package-relative resolution (primary)
- **Level 2**: Return original path (fallback)
- **Level 3**: Existing absolute path behavior (unchanged)

### Implementation Impact

#### **Minimal Code Changes**
- **1 method replacement** in `BasePipelineConfig._convert_to_relative_path()`
- **2 utility functions** added to handle package-relative path operations
- **0 changes required** in step builders (they continue using existing APIs)

#### **Zero Breaking Changes**
- All existing configurations continue working
- All existing step builders continue working
- All existing APIs remain unchanged
- Gradual improvement without disruption

#### **Immediate Problem Resolution**
- Lambda deployment path resolution failures: **FIXED**
- Container deployment path issues: **FIXED**
- PyPI package installation path problems: **FIXED**
- Development environment compatibility: **MAINTAINED**

## Implementation Strategy

### Single-Day Implementation (Minimal Viable Solution)

Based on the code redundancy evaluation guide principles, this implementation can be completed in **one focused development session** because it only requires **two utility functions** and **one method replacement**.

#### **Step 1: Add Utility Functions (30 minutes)**
Create `src/cursus/core/utils/path_resolution.py`:

```python
"""Deployment-context-agnostic path resolution utilities."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_package_relative_path(absolute_path: str) -> str:
    """Convert absolute path to package-relative path for deployment portability."""
    # Implementation as shown in architecture section
    
def resolve_package_relative_path(relative_path: str) -> str:
    """Resolve package-relative path to absolute path in current deployment context."""
    # Implementation as shown in architecture section
```

#### **Step 2: Update Base Configuration (15 minutes)**
Modify `src/cursus/core/base/config_base.py`:

```python
# Add import
from ..utils.path_resolution import get_package_relative_path, resolve_package_relative_path

class BasePipelineConfig(BaseModel, ABC):
    # ... existing code unchanged ...
    
    def _convert_to_relative_path(self, path: str) -> str:
        """Convert absolute path to package-relative path (REPLACEMENT METHOD)."""
        return get_package_relative_path(path)
    
    def get_resolved_path(self, relative_path: str) -> str:
        """Resolve package-relative path to absolute path (NEW HELPER METHOD)."""
        return resolve_package_relative_path(relative_path)
```

#### **Step 3: Add Basic Tests (15 minutes)**
Create `test/core/utils/test_path_resolution.py`:

```python
"""Tests for path resolution utilities."""

import pytest
from pathlib import Path
from cursus.core.utils.path_resolution import get_package_relative_path, resolve_package_relative_path

def test_get_package_relative_path():
    """Test package-relative path conversion."""
    # Test cases for different path patterns
    
def test_resolve_package_relative_path():
    """Test package-relative path resolution."""
    # Test cases for path resolution
```

#### **Total Implementation Time: ~1 Hour**

### Why This Minimal Approach Works

#### ✅ **Addresses Root Cause Immediately**
- Replaces problematic `Path.cwd()` usage with package-aware resolution
- Fixes Lambda deployment failures with minimal code change
- No complex architecture needed for this simple problem

#### ✅ **Zero Risk Implementation**
- **No breaking changes** - existing APIs unchanged
- **Fallback behavior** - returns original path if conversion fails
- **Gradual rollout** - can be deployed incrementally

#### ✅ **Follows YAGNI Principle**
- **You Aren't Gonna Need It** - no speculative features
- **Solves actual problem** - addresses validated Lambda deployment issue
- **Simple maintenance** - easy to understand and modify

### Deployment Strategy

#### **Phase 1: Internal Testing (Same Day)**
- Deploy to development environment
- Test with existing configurations
- Validate Lambda deployment scenarios

#### **Phase 2: Gradual Rollout (Next Day)**
- Deploy to staging environment
- Monitor path resolution success rates
- Validate across different deployment contexts

#### **Phase 3: Production Deployment (Day 3)**
- Deploy to production with monitoring
- Track performance metrics
- Monitor for any fallback usage

### Success Metrics

#### **Immediate Success Indicators**
- Lambda deployment path resolution: **WORKING**
- Existing configurations: **UNCHANGED BEHAVIOR**
- Performance impact: **NEGLIGIBLE**
- Code complexity: **REDUCED**

#### **Long-term Success Indicators**
- Zero deployment context path failures
- Improved developer experience with portable configurations
- Reduced support tickets for path-related issues
- Simplified debugging and troubleshooting

## Configuration File Format Evolution

### Current Format (Problematic)
```json
{
  "configuration": {
    "shared": {
      "source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz",
      "portable_source_dir": "dockers/xgboost_atoz"
    },
    "specific": {
      "TabularPreprocessing_training": {
        "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts",
        "portable_processing_source_dir": "dockers/xgboost_atoz/scripts",
        "processing_entry_point": "tabular_preprocessing.py",
        "script_path": "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
      },
      "ModelCalibration_calibration": {
        "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts",
        "portable_processing_source_dir": "dockers/xgboost_atoz/scripts",
        "processing_entry_point": "model_calibration.py",
        "script_path": "dockers/xgboost_atoz/scripts/model_calibration.py"
      },
      "XGBoostModelEval_calibration": {
        "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz",
        "portable_processing_source_dir": "dockers/xgboost_atoz",
        "processing_entry_point": "xgboost_model_evaluation.py",
        "script_path": "dockers/xgboost_atoz/xgboost_model_evaluation.py"
      }
    }
  }
}
```

### Enhanced Runtime Behavior (Same Configuration Format)

**The configuration file format remains exactly the same** - we only change how the portable paths are resolved at runtime:

```json
{
  "configuration": {
    "shared": {
      "source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz",
      "portable_source_dir": "dockers/xgboost_atoz"
    },
    "specific": {
      "TabularPreprocessing_training": {
        "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts",
        "portable_processing_source_dir": "dockers/xgboost_atoz/scripts",
        "processing_entry_point": "tabular_preprocessing.py",
        "script_path": "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
      }
    }
  }
}
```

**What Changes: Runtime Path Resolution Behavior**

| Deployment Context | Current Behavior (Problematic) | Enhanced Behavior (Fixed) |
|-------------------|--------------------------------|---------------------------|
| **Development** | `portable_source_dir` → `/workspace/project/dockers/xgboost_atoz` | `portable_source_dir` → `/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz` |
| **Lambda** | `portable_source_dir` → `/var/task/dockers/xgboost_atoz` ❌ **FAILS** | `portable_source_dir` → `/tmp/cursus/dockers/xgboost_atoz` ✅ **WORKS** |
| **Container** | `portable_source_dir` → `/app/dockers/xgboost_atoz` ❌ **FAILS** | `portable_source_dir` → `/usr/local/lib/python3.x/site-packages/cursus/dockers/xgboost_atoz` ✅ **WORKS** |

**Key Enhancement: Package-Aware Resolution**

Instead of resolving `portable_source_dir` relative to `Path.cwd()` (current working directory), the enhanced system resolves it relative to the **package installation location**:

```python
# CURRENT (Problematic): Working directory relative
def _convert_to_relative_path(self, path: str) -> str:
    runtime_location = Path.cwd()  # /var/task/ in Lambda, /app/ in container
    # Results in paths that don't exist in deployment contexts

# ENHANCED (Fixed): Package-aware relative  
def _convert_to_relative_path(self, path: str) -> str:
    return get_package_relative_path(path)  # Always relative to package location
    # Results in paths that work across all deployment contexts
```

**Runtime Resolution Examples:**

```python
# Configuration contains: "portable_processing_source_dir": "dockers/xgboost_atoz/scripts"

# Development Environment:
# Package root: /home/ec2-user/SageMaker/Cursus/
# Resolved path: /home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts ✅

# Lambda Environment:  
# Package root: /tmp/cursus/
# Resolved path: /tmp/cursus/dockers/xgboost_atoz/scripts ✅

# Container Environment:
# Package root: /usr/local/lib/python3.x/site-packages/cursus/
# Resolved path: /usr/local/lib/python3.x/site-packages/cursus/dockers/xgboost_atoz/scripts ✅
```

**Zero Configuration Changes Required:**
- Same JSON configuration files work everywhere
- Same portable path values (`"dockers/xgboost_atoz/scripts"`)
- Only the runtime resolution logic changes
- Complete backward compatibility maintained

### Migration Strategy

#### Automatic Migration
- Existing configurations automatically enhanced with package-relative paths
- Original absolute paths preserved for backward compatibility
- Gradual migration to package-relative-only format over time

#### Validation and Fallback
- Runtime validation of package-relative paths
- Automatic fallback to absolute paths when package-relative resolution fails
- Comprehensive logging for migration debugging

## Deployment Context Compatibility

### Development Environment
- **Package Detection**: Source directory structure recognition
- **Path Resolution**: Direct file system access with package-relative conversion
- **Optimization**: Caching of package root discovery for performance

### PyPI Package Installation
- **Package Detection**: Site-packages directory recognition
- **Path Resolution**: Package-relative paths resolved to installation location
- **Optimization**: Importlib.metadata integration for efficient package location

### Serverless Deployment (Lambda, Cloud Functions)
- **Package Detection**: Temporary directory package installation recognition
- **Path Resolution**: Package-relative paths resolved to temporary installation location
- **Optimization**: Environment variable-based context detection for fast identification

### Container Deployment (Docker, Kubernetes)
- **Package Detection**: Container-specific package installation patterns
- **Path Resolution**: Package-relative paths resolved to container installation location
- **Optimization**: Container environment indicator-based context detection

## Error Handling and Fallback Mechanisms

### Graceful Degradation Strategy

#### Level 1: Package-Relative Resolution
- Primary strategy using package-aware path resolution
- Highest portability and reliability across deployment contexts

#### Level 2: Working Directory Fallback
- Fallback to current working directory relative paths
- Maintains compatibility with existing behavior when package resolution fails

#### Level 3: Absolute Path Preservation
- Final fallback to original absolute paths
- Ensures system continues functioning even when all relative resolution fails

### Error Recovery Mechanisms

```python
class PathResolutionError(Exception):
    """Base exception for path resolution failures."""
    pass

class PackageRootNotFoundError(PathResolutionError):
    """Raised when package root cannot be determined."""
    pass

class PathConversionError(PathResolutionError):
    """Raised when path conversion fails."""
    pass

def resolve_path_with_fallback(self, path: str) -> str:
    """Resolve path with comprehensive fallback strategy."""
    try:
        # Level 1: Package-relative resolution
        return self._resolve_package_relative(path)
    except PackageRootNotFoundError:
        logger.warning("Package root not found, falling back to working directory resolution")
        try:
            # Level 2: Working directory fallback
            return self._resolve_working_directory_relative(path)
        except PathConversionError:
            logger.warning("Working directory resolution failed, using absolute path")
            # Level 3: Absolute path preservation
            return path
    except Exception as e:
        logger.error(f"Unexpected error in path resolution: {e}")
        return path
```

## Performance Considerations

### Caching Strategy

#### Package Root Caching
- Cache package root discovery results per process
- Invalidate cache only when package structure changes
- Memory-efficient caching with weak references

#### Path Resolution Caching
- Cache frequently accessed path conversions
- LRU cache with configurable size limits
- Context-aware cache invalidation

### Optimization Techniques

#### Lazy Initialization
- Initialize path resolver only when needed
- Defer expensive operations until first use
- Minimize startup time impact

#### Batch Processing
- Process multiple paths in single operations
- Reduce repeated package root discovery
- Optimize for configuration loading scenarios

## Security Considerations

### Path Traversal Prevention
- Validate that resolved paths remain within package boundaries
- Prevent directory traversal attacks through malicious path inputs
- Sanitize path components before resolution

### Access Control
- Respect file system permissions in all deployment contexts
- Fail gracefully when paths are inaccessible
- Log security-related path resolution failures

## Monitoring and Observability

### Metrics Collection
- Path resolution success/failure rates
- Performance metrics for path resolution operations
- Deployment context detection accuracy

### Logging Strategy
- Structured logging for path resolution operations
- Debug-level logging for troubleshooting
- Warning-level logging for fallback usage

### Health Checks
- Validate package root accessibility
- Monitor path resolution performance
- Alert on high fallback usage rates

## Future Extensions

### Advanced Deployment Context Support
- Cloud-specific optimizations (AWS, GCP, Azure)
- Edge computing environment support
- Hybrid deployment scenario handling

### Configuration Management Integration
- Integration with configuration management systems
- Dynamic path resolution based on deployment metadata
- Environment-specific path overrides

### Performance Optimizations
- Parallel path resolution for large configurations
- Predictive caching based on usage patterns
- Memory-mapped file access for large packages

## Migration Guide

### For Existing Configurations
1. **No Immediate Changes Required**: Existing configurations continue working with automatic enhancement
2. **Gradual Migration**: Optionally migrate to package-relative-only format over time
3. **Validation Tools**: Use provided tools to validate configuration portability

### For New Configurations
1. **Use Enhanced Base Classes**: Inherit from `DeploymentAgnosticConfigBase`
2. **Leverage Package-Relative Paths**: Use portable path properties for maximum portability
3. **Test Across Contexts**: Validate configurations in target deployment environments

### For Step Builders
1. **Minimal Changes**: Update to use resolved paths from enhanced configurations
2. **Automatic Fallback**: Benefit from automatic fallback mechanisms
3. **Enhanced Logging**: Leverage improved logging for debugging

## Critical Design Error Discovery and Resolution

### **❌ CRITICAL ERROR IN ORIGINAL DESIGN**

**IMPORTANT UPDATE (2025-09-22)**: During implementation and testing of this design, a **critical architectural error** was discovered that rendered the original approach insufficient for Lambda deployments. The error was so fundamental that it required a complete redesign of the path resolution strategy.

**Original Flawed Assumption**: The design assumed that in all deployment contexts, target files would be **children** of the cursus package directory. This assumption was fundamentally incorrect for Lambda deployments.

**Actual Lambda Architecture Discovered**: Lambda deployments use a **sibling directory structure** where cursus and target files are siblings, not parent-child:

```
❌ ASSUMED (WRONG):
/tmp/buyer_abuse_mods_template/cursus/
└── mods_pipeline_adapter/          # Assumed child directory
    └── dockers/
        └── xgboost_atoz/
            └── scripts/
                └── tabular_preprocessing.py

✅ ACTUAL (CORRECT):
/tmp/buyer_abuse_mods_template/
├── cursus/                         # cursus package
│   ├── __init__.py
│   └── ...
└── mods_pipeline_adapter/          # SIBLING directory
    └── dockers/
        └── xgboost_atoz/
            └── scripts/
                └── tabular_preprocessing.py
```

**Impact of This Error**:
- **MODS Lambda Deployment Failures**: Path resolution failed because files were looked for in wrong location
- **Universal Portability Broken**: Lambda deployments couldn't use the same configs as other environments
- **Critical Production Issue**: Real-world Lambda deployments would fail with "file not found" errors

### **✅ CORRECTED DESIGN IMPLEMENTATION**

The error was resolved through enhanced implementation documented in:

**📋 [2025-09-22 MODS Lambda Sibling Directory Path Resolution Fix - Implementation Completion](../2_project_planning/2025-09-22_mods_lambda_sibling_directory_path_resolution_fix_completion.md)**

**Key Corrections**:
1. **Multi-Strategy Path Resolution**: Try child path first, then sibling path, with fallback
2. **Realistic Testing**: Proper simulation of runtime vs development time separation
3. **Universal Compatibility**: Verified across all deployment contexts including Lambda's unique sibling structure

**Enhanced Resolution Algorithm**:
```python
def resolve_package_relative_path(relative_path: str) -> str:
    cursus_package_dir = Path(cursus.__file__).parent
    
    # Strategy 1: Try as child (traditional structure)
    child_path = cursus_package_dir / relative_path
    if child_path.exists():
        return str(child_path.resolve())
    
    # Strategy 2: Try as sibling (Lambda structure)
    sibling_path = cursus_package_dir.parent / relative_path
    if sibling_path.exists():
        return str(sibling_path.resolve())
    
    # Strategy 3: Fallback for backward compatibility
    return str(child_path.resolve())
```

## References

### Design Documents
- **[Config Portability Path Resolution Design](./config_portability_path_resolution_design.md)** - Original design document for portable path resolution
- **[Cursus Package Portability Architecture Design](./cursus_package_portability_architecture_design.md)** - Overall package portability architecture

### Implementation Plans
- **[Config Portability Path Resolution Implementation Plan](../2_project_planning/2025-09-20_config_portability_path_resolution_implementation_plan.md)** - Detailed implementation roadmap
- **[Pipeline Execution Temp Dir Implementation Plan](../2_project_planning/2025-09-18_pipeline_execution_temp_dir_implementation_plan.md)** - Related temporary directory handling

### Analysis Documents
- **[MODS Pipeline Path Resolution Error Analysis](../.internal/mods_pipeline_path_resolution_error_analysis.md)** - Comprehensive error analysis that motivated this design
- **[MODS Deployment Process Reconstruction](../.internal/mods_deployment_process_reconstruction.md)** - Deployment architecture analysis

### Configuration System Documentation
- **[Config Tiered Design](./config_tiered_design.md)** - Three-tier configuration architecture
- **[Config Field Categorization Consolidated](./config_field_categorization_consolidated.md)** - Configuration field management principles
- **[Unified Step Catalog System Design](./unified_step_catalog_system_design.md)** - Step catalog integration architecture
