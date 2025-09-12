---
tags:
  - project
  - planning
  - runtime_testing
  - dependency_resolution
  - automation
keywords:
  - automatic dependency resolution
  - script runtime testing
  - dependency management
  - virtual environments
  - package installation
  - import error handling
  - developer experience
  - testing automation
topics:
  - runtime testing enhancement
  - dependency automation
  - developer productivity
  - testing infrastructure
language: python
date of note: 2025-09-12
---

# Automatic Dependency Resolution for Runtime Testing

## Project Overview

This project aims to implement automatic dependency resolution for the Cursus runtime testing system, addressing the most common pain point developers face: script import/dependency errors during testing. The system will automatically detect, resolve, and install missing dependencies when testing pipeline scripts.

## Problem Statement

### Current Pain Points
1. **Manual Dependency Management**: Developers must manually identify and install script dependencies
2. **Import Errors**: Scripts fail with `ModuleNotFoundError` for packages like pandas, xgboost, sklearn, boto3
3. **Environment Conflicts**: Different scripts may require different package versions
4. **Time-Consuming Debugging**: Developers spend significant time diagnosing and fixing dependency issues
5. **Inconsistent Environments**: Testing environments may differ from production environments

### Impact Assessment
- **Developer Productivity**: 30-60 minutes per script for dependency troubleshooting
- **Testing Reliability**: 40% of script test failures are dependency-related
- **Onboarding Friction**: New developers struggle with environment setup
- **CI/CD Reliability**: Automated testing fails due to missing dependencies

## Solution Architecture

### Core Components

#### 1. DependencyResolver
**Purpose**: Main orchestrator for automatic dependency resolution
**Location**: `src/cursus/validation/runtime/dependency_resolver.py`

```python
class DependencyResolver:
    """Automatic dependency resolution for script runtime testing."""
    
    def __init__(self, cache_dir: str, install_timeout: int = 300):
        self.cache_dir = Path(cache_dir)
        self.install_timeout = install_timeout
        self.analyzer = ScriptDependencyAnalyzer()
        self.mapper = PackageMapper()
        self.installer = DependencyInstaller()
        self.env_manager = VirtualEnvironmentManager()
```

#### 2. ScriptDependencyAnalyzer
**Purpose**: Analyze scripts to extract dependency information
**Location**: `src/cursus/validation/runtime/script_dependency_analyzer.py`

**Key Features**:
- Parse Python AST to find import statements
- Extract both direct imports and from-imports
- Handle dynamic imports and conditional imports
- Build dependency graph for complex scripts
- Detect version requirements from comments/docstrings

#### 3. PackageMapper
**Purpose**: Map import names to correct PyPI package names
**Location**: `src/cursus/validation/runtime/package_mapper.py`

**Key Features**:
- Comprehensive mapping database (sklearn → scikit-learn, cv2 → opencv-python)
- Version compatibility checking
- Alternative package suggestions
- Custom mapping configuration support

#### 4. VirtualEnvironmentManager
**Purpose**: Manage isolated testing environments
**Location**: `src/cursus/validation/runtime/virtual_env_manager.py`

**Key Features**:
- Create isolated virtual environments per script/test
- Environment caching and reuse
- Cleanup and garbage collection
- Environment inheritance and layering

#### 5. DependencyInstaller
**Purpose**: Handle automatic package installation
**Location**: `src/cursus/validation/runtime/dependency_installer.py`

**Key Features**:
- Pip-based installation with timeout handling
- Batch installation optimization
- Installation verification
- Rollback capability on failure

#### 6. DependencyCache
**Purpose**: Cache resolved dependencies to avoid repeated work
**Location**: `src/cursus/validation/runtime/dependency_cache.py`

**Key Features**:
- Persistent cache storage
- Cache invalidation strategies
- Environment fingerprinting
- Cache sharing across test runs

### Integration Points

#### Enhanced RuntimeTester
**File**: `src/cursus/validation/runtime/runtime_testing.py`

```python
class RuntimeTester:
    def __init__(self, workspace_dir, auto_resolve_dependencies=True, 
                 dependency_resolver_config=None):
        # Existing initialization...
        self.auto_resolve_dependencies = auto_resolve_dependencies
        if auto_resolve_dependencies:
            self.dependency_resolver = DependencyResolver(
                cache_dir=f"{workspace_dir}/.dependency_cache",
                **dependency_resolver_config or {}
            )
    
    def test_script_with_spec(self, script_spec, main_params):
        """Enhanced with automatic dependency resolution."""
        if self.auto_resolve_dependencies:
            # Resolve dependencies before testing
            resolution_result = self.dependency_resolver.resolve_for_script(
                script_spec.script_name, script_spec.script_path
            )
            if not resolution_result.success:
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message=f"Dependency resolution failed: {resolution_result.error}",
                    execution_time=0.0,
                    has_main_function=False
                )
        
        # Proceed with existing testing logic...
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
**Deliverables**:
- [ ] ScriptDependencyAnalyzer implementation
- [ ] PackageMapper with comprehensive mapping database
- [ ] Basic DependencyResolver orchestration
- [ ] Unit tests for core components

**Key Tasks**:
1. Implement AST-based import analysis
2. Create package mapping database with common ML/data packages
3. Design resolver API and error handling
4. Set up testing infrastructure

### Phase 2: Environment Management (Week 3)
**Deliverables**:
- [ ] VirtualEnvironmentManager implementation
- [ ] Environment caching and reuse logic
- [ ] Environment cleanup and garbage collection
- [ ] Integration tests with virtual environments

**Key Tasks**:
1. Implement virtual environment creation and management
2. Design environment caching strategy
3. Implement cleanup and resource management
4. Test environment isolation and inheritance

### Phase 3: Installation Engine (Week 4)
**Deliverables**:
- [ ] DependencyInstaller implementation
- [ ] Batch installation optimization
- [ ] Installation verification and rollback
- [ ] Error handling and retry logic

**Key Tasks**:
1. Implement pip-based installation with proper error handling
2. Add batch installation for efficiency
3. Implement installation verification
4. Add rollback capability for failed installations

### Phase 4: Caching System (Week 5)
**Deliverables**:
- [ ] DependencyCache implementation
- [ ] Persistent cache storage
- [ ] Cache invalidation strategies
- [ ] Performance optimization

**Key Tasks**:
1. Design cache storage format and location
2. Implement cache invalidation based on script changes
3. Add cache sharing and synchronization
4. Optimize cache performance and storage

### Phase 5: RuntimeTester Integration (Week 6)
**Deliverables**:
- [ ] Enhanced RuntimeTester with auto-resolution
- [ ] Configuration options and safety controls
- [ ] Backward compatibility maintenance
- [ ] Integration tests

**Key Tasks**:
1. Integrate DependencyResolver into RuntimeTester
2. Add configuration options for control and safety
3. Ensure backward compatibility
4. Comprehensive integration testing

### Phase 6: User Experience & Safety (Week 7)
**Deliverables**:
- [ ] User approval prompts and dry-run mode
- [ ] Package whitelist/blacklist functionality
- [ ] Detailed logging and progress reporting
- [ ] Error recovery and fallback mechanisms

**Key Tasks**:
1. Implement user approval workflows
2. Add safety controls and package filtering
3. Enhance logging and user feedback
4. Implement graceful error handling

### Phase 7: Documentation & Testing (Week 8)
**Deliverables**:
- [ ] Updated tutorials with auto-resolution examples
- [ ] API documentation for new components
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

**Key Tasks**:
1. Update runtime testing tutorials
2. Create API documentation
3. Comprehensive testing across different scenarios
4. Performance testing and optimization

## Technical Specifications

### Configuration Schema

```python
@dataclass
class DependencyResolverConfig:
    """Configuration for automatic dependency resolution."""
    
    # Core settings
    enabled: bool = True
    cache_dir: Optional[str] = None
    install_timeout: int = 300  # 5 minutes
    
    # Safety controls
    require_approval: bool = False
    dry_run_mode: bool = False
    package_whitelist: Optional[List[str]] = None
    package_blacklist: Optional[List[str]] = None
    
    # Environment settings
    use_virtual_env: bool = True
    env_cache_ttl: int = 86400  # 24 hours
    max_cached_envs: int = 10
    
    # Installation settings
    pip_extra_args: List[str] = field(default_factory=list)
    allow_pre_release: bool = False
    prefer_binary: bool = True
    
    # Logging and feedback
    verbose: bool = True
    progress_callback: Optional[Callable] = None
```

### API Design

```python
# Simple usage - auto-resolution enabled by default
tester = RuntimeTester("./test_workspace", auto_resolve_dependencies=True)

# Advanced configuration
config = DependencyResolverConfig(
    require_approval=True,
    package_whitelist=["pandas", "numpy", "xgboost", "scikit-learn"],
    verbose=True
)
tester = RuntimeTester("./test_workspace", 
                      auto_resolve_dependencies=True,
                      dependency_resolver_config=config)

# Test script - dependencies resolved automatically
result = tester.test_script_with_spec(spec, main_params)
```

### Error Handling Strategy

```python
class DependencyResolutionResult:
    """Result of dependency resolution attempt."""
    
    success: bool
    resolved_packages: List[str]
    failed_packages: List[str]
    warnings: List[str]
    execution_time: float
    cache_hit: bool
    environment_path: Optional[str]
    error_details: Optional[str]
```

## Safety and Security Considerations

### Package Security
- **Whitelist Approach**: Default to allowing only well-known, trusted packages
- **Signature Verification**: Verify package signatures when possible
- **Version Pinning**: Prefer specific versions over latest
- **Vulnerability Scanning**: Integration with security scanning tools

### Environment Isolation
- **Strict Isolation**: Each test runs in isolated virtual environment
- **Resource Limits**: CPU and memory limits for installation processes
- **Network Restrictions**: Limit network access during testing
- **Cleanup Guarantees**: Ensure environments are properly cleaned up

### User Control
- **Explicit Approval**: Option to require user approval for installations
- **Dry Run Mode**: Show what would be installed without actually installing
- **Audit Logging**: Log all dependency resolution activities
- **Rollback Capability**: Ability to revert installations

## Performance Considerations

### Optimization Strategies
- **Dependency Caching**: Cache resolved dependencies across test runs
- **Environment Reuse**: Reuse compatible virtual environments
- **Batch Installation**: Install multiple packages in single pip call
- **Parallel Processing**: Resolve dependencies for multiple scripts in parallel

### Performance Targets
- **Cold Start**: < 2 minutes for first-time dependency resolution
- **Warm Start**: < 30 seconds for cached dependencies
- **Memory Usage**: < 500MB additional memory overhead
- **Storage**: < 1GB cache storage per workspace

## Testing Strategy

### Unit Tests
- [ ] ScriptDependencyAnalyzer: AST parsing and import extraction
- [ ] PackageMapper: Import name to package name mapping
- [ ] VirtualEnvironmentManager: Environment creation and management
- [ ] DependencyInstaller: Package installation and verification
- [ ] DependencyCache: Cache operations and invalidation

### Integration Tests
- [ ] End-to-end dependency resolution workflow
- [ ] RuntimeTester integration with auto-resolution
- [ ] Multi-script pipeline testing with dependencies
- [ ] Error handling and recovery scenarios
- [ ] Performance testing with large dependency sets

### Compatibility Tests
- [ ] Python version compatibility (3.8, 3.9, 3.10, 3.11)
- [ ] Operating system compatibility (Linux, macOS, Windows)
- [ ] Package manager compatibility (pip, conda)
- [ ] Virtual environment compatibility (venv, virtualenv, conda)

## Monitoring and Metrics

### Key Metrics
- **Resolution Success Rate**: Percentage of successful dependency resolutions
- **Resolution Time**: Average time to resolve dependencies
- **Cache Hit Rate**: Percentage of cache hits vs. misses
- **Installation Failure Rate**: Percentage of failed package installations
- **User Satisfaction**: Developer feedback on auto-resolution experience

### Monitoring Implementation
- **Structured Logging**: JSON-formatted logs for analysis
- **Metrics Collection**: Integration with monitoring systems
- **Error Tracking**: Detailed error reporting and categorization
- **Performance Profiling**: Regular performance analysis

## Migration and Rollout Plan

### Phase 1: Opt-in Beta (Week 9-10)
- Deploy with `auto_resolve_dependencies=False` by default
- Allow developers to opt-in for testing
- Collect feedback and performance data
- Fix critical issues and edge cases

### Phase 2: Gradual Rollout (Week 11-12)
- Enable by default for new workspaces
- Provide clear migration guide for existing workspaces
- Monitor adoption and success rates
- Address compatibility issues

### Phase 3: Full Deployment (Week 13-14)
- Enable by default for all workspaces
- Provide fallback mechanisms for edge cases
- Complete documentation and training materials
- Establish support processes

## Risk Assessment and Mitigation

### Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Package installation failures | High | Medium | Robust error handling, fallback mechanisms |
| Environment conflicts | Medium | Low | Strict environment isolation |
| Performance degradation | Medium | Low | Caching, optimization, performance monitoring |
| Security vulnerabilities | High | Low | Package whitelisting, security scanning |

### Operational Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Developer resistance | Medium | Medium | Clear benefits communication, opt-in approach |
| Increased support burden | Medium | Medium | Comprehensive documentation, self-service tools |
| CI/CD integration issues | High | Low | Thorough testing, gradual rollout |

## Success Criteria

### Quantitative Metrics
- [ ] 90% reduction in dependency-related test failures
- [ ] 80% reduction in time spent on dependency troubleshooting
- [ ] 95% dependency resolution success rate
- [ ] < 2 minute average resolution time for new dependencies
- [ ] 80% cache hit rate for repeated tests

### Qualitative Metrics
- [ ] Positive developer feedback on ease of use
- [ ] Reduced support tickets related to dependency issues
- [ ] Improved onboarding experience for new developers
- [ ] Increased adoption of runtime testing

## Future Enhancements

### Phase 2 Features (Future)
- **Conda Integration**: Support for conda package manager
- **Docker Integration**: Container-based dependency isolation
- **Dependency Optimization**: Automatic dependency conflict resolution
- **ML Model Dependencies**: Special handling for ML model files and weights
- **Cloud Integration**: Integration with cloud-based package repositories

### Advanced Features
- **Predictive Caching**: Pre-cache dependencies based on usage patterns
- **Dependency Analytics**: Analyze dependency usage across the organization
- **Security Scanning**: Automatic vulnerability scanning of dependencies
- **License Compliance**: Automatic license compatibility checking

## References and Related Work

### Internal Documentation
- **[Runtime Testing Quick Start Guide](../5_tutorials/validation/script_runtime_tester_quick_start.md)** - Current runtime testing documentation with manual dependency management
- **[Runtime Testing API Reference](../5_tutorials/validation/script_runtime_tester_api_reference.md)** - Complete API documentation for runtime testing system
- **[Script Development Guide](../0_developer_guide/script_development_guide.md)** - Guidelines for script development and dependency management
- **[Validation Framework Guide](../0_developer_guide/validation_framework_guide.md)** - Overall validation framework architecture

### Design Documents
- **[Pipeline Runtime Testing Simplified Design](../1_design/pipeline_runtime_testing_simplified_design.md)** - Core design document for the runtime testing system
- **[Contract Discovery Manager Design](../1_design/contract_discovery_manager_design.md)** - Design for automatic contract discovery that could inform dependency resolution
- **[Dependency Resolution System](../1_design/dependency_resolution_system.md)** - General dependency resolution architecture (if exists)

### External References
- **[Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)** - Official Python documentation on virtual environments
- **[Pip Documentation](https://pip.pypa.io/en/stable/)** - Package installation and management
- **[AST Module Documentation](https://docs.python.org/3/library/ast.html)** - Python AST parsing for import analysis
- **[Packaging Best Practices](https://packaging.python.org/en/latest/)** - Python packaging and dependency management best practices

### Similar Tools and Inspiration
- **[Pipenv](https://pipenv.pypa.io/en/latest/)** - Python dependency management with automatic virtual environments
- **[Poetry](https://python-poetry.org/)** - Modern dependency management and packaging tool
- **[Conda](https://docs.conda.io/en/latest/)** - Package and environment management system
- **[Docker](https://www.docker.com/)** - Containerization for dependency isolation

## Conclusion

The automatic dependency resolution system will significantly improve the developer experience with runtime testing by eliminating the most common source of test failures and debugging time. The phased implementation approach ensures safety and reliability while providing immediate value to developers.

The system's design prioritizes safety, performance, and user control while providing intelligent automation that adapts to the diverse dependency requirements of pipeline scripts. With proper implementation and rollout, this enhancement will make runtime testing more accessible and reliable for all developers.

## Appendix

### A. Package Mapping Database Schema

```python
PACKAGE_MAPPINGS = {
    # ML and Data Science
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    
    # AWS and Cloud
    "boto3": "boto3",
    "botocore": "botocore",
    "sagemaker": "sagemaker",
    
    # Common aliases
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib",
    "sns": "seaborn",
    
    # Version-specific mappings
    "tensorflow": {
        "default": "tensorflow>=2.8.0",
        "gpu": "tensorflow-gpu>=2.8.0",
        "cpu": "tensorflow-cpu>=2.8.0"
    }
}
```

### B. Environment Configuration Template

```python
ENVIRONMENT_TEMPLATE = {
    "python_version": "3.9",
    "base_packages": [
        "pip>=21.0",
        "setuptools>=60.0",
        "wheel>=0.37.0"
    ],
    "common_ml_packages": [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0"
    ],
    "aws_packages": [
        "boto3>=1.20.0",
        "sagemaker>=2.100.0"
    ]
}
```

### C. Error Code Reference

```python
class DependencyResolutionError(Exception):
    """Base exception for dependency resolution errors."""
    
    ERROR_CODES = {
        "ANALYSIS_FAILED": "Failed to analyze script dependencies",
        "PACKAGE_NOT_FOUND": "Package not found in any repository",
        "INSTALLATION_FAILED": "Package installation failed",
        "ENVIRONMENT_CREATION_FAILED": "Virtual environment creation failed",
        "TIMEOUT": "Dependency resolution timed out",
        "PERMISSION_DENIED": "Insufficient permissions for installation",
        "NETWORK_ERROR": "Network error during package download",
        "CACHE_ERROR": "Dependency cache operation failed"
    }
