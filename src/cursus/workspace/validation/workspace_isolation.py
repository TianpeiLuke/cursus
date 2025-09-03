"""
Test Workspace Isolation

Provides advanced test workspace isolation utilities, integrating with the
Phase 1 consolidated workspace isolation system. This module extends the
core isolation capabilities with test-specific isolation requirements.

Architecture Integration:
- Extends Phase 1 WorkspaceIsolationManager for test-specific isolation
- Provides test environment sandboxing and boundary enforcement
- Integrates with test workspace management system
- Coordinates with existing validation frameworks

Features:
- Test environment isolation validation
- Test data and output isolation
- Test dependency isolation
- Integration with Phase 1 isolation manager
- Advanced test boundary enforcement
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, Set
import logging
from datetime import datetime
import tempfile
import shutil

from pydantic import BaseModel, Field, ConfigDict

# PHASE 3 INTEGRATION: Import Phase 1 consolidated isolation system
from ..core.isolation import WorkspaceIsolationManager, IsolationViolation
from ..core.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class WorkspaceIsolationConfig(BaseModel):
    """Configuration for workspace isolation."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    enable_path_isolation: bool = True
    enable_environment_isolation: bool = True
    enable_dependency_isolation: bool = True
    enable_resource_isolation: bool = True
    allowed_system_paths: List[str] = Field(default_factory=lambda: ["/usr", "/bin", "/lib"])
    blocked_paths: List[str] = Field(default_factory=list)
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    timeout_seconds: Optional[int] = None


class IsolationEnvironment(BaseModel):
    """Represents an isolated test environment."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    environment_id: str
    test_workspace_path: str
    isolation_config: WorkspaceIsolationConfig
    python_path: List[str] = Field(default_factory=list)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # "active", "suspended", "terminated"


class WorkspaceTestIsolationManager:
    """
    Advanced test workspace isolation manager integrating with Phase 1.
    
    This manager extends the Phase 1 WorkspaceIsolationManager with test-specific
    isolation capabilities. It provides:
    - Test environment sandboxing
    - Test data and output isolation
    - Test dependency isolation
    - Advanced boundary enforcement
    - Integration with Phase 1 isolation system
    
    Phase 3 Integration Features:
    - Extends Phase 1 WorkspaceIsolationManager
    - Leverages core isolation validation capabilities
    - Adds test-specific isolation requirements
    - Provides advanced test environment sandboxing
    """
    
    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
        isolation_config: Optional[WorkspaceIsolationConfig] = None
    ):
        """
        Initialize test workspace isolation manager.
        
        Args:
            workspace_manager: Phase 1 consolidated workspace manager
            isolation_config: Test-specific isolation configuration
        """
        # PHASE 3 INTEGRATION: Use Phase 1 workspace manager
        if workspace_manager:
            self.workspace_manager = workspace_manager
            self.core_isolation_manager = workspace_manager.isolation_manager
        else:
            from ..core.manager import WorkspaceManager
            self.workspace_manager = WorkspaceManager()
            self.core_isolation_manager = self.workspace_manager.isolation_manager
        
        # Test-specific isolation configuration
        self.isolation_config = isolation_config or WorkspaceIsolationConfig()
        
        # Active isolated environments
        self.isolated_environments: Dict[str, IsolationEnvironment] = {}
        
        logger.info("Initialized test workspace isolation manager with Phase 1 integration")
    
    # Test Environment Isolation
    
    def create_isolated_test_environment(
        self,
        test_workspace_path: str,
        environment_id: Optional[str] = None,
        **kwargs
    ) -> IsolationEnvironment:
        """
        Create an isolated test environment.
        
        Args:
            test_workspace_path: Path to test workspace
            environment_id: Optional environment identifier
            **kwargs: Additional isolation configuration
        
        Returns:
            IsolationEnvironment configuration
        """
        if not environment_id:
            environment_id = f"test_env_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating isolated test environment: {environment_id}")
        
        try:
            # Validate test workspace path
            workspace_path = Path(test_workspace_path)
            if not workspace_path.exists():
                raise ValueError(f"Test workspace path does not exist: {test_workspace_path}")
            
            # Use Phase 1 isolation manager for basic isolation setup
            core_isolation_result = self.core_isolation_manager.create_isolated_environment(
                workspace_id=environment_id
            )
            
            # Create test-specific isolation configuration
            test_isolation_config = WorkspaceIsolationConfig(**kwargs) if kwargs else self.isolation_config
            
            # Set up test-specific isolation
            isolated_env = IsolationEnvironment(
                environment_id=environment_id,
                test_workspace_path=str(workspace_path),
                isolation_config=test_isolation_config
            )
            
            # Configure test environment isolation
            self._configure_test_isolation(isolated_env, core_isolation_result)
            
            # Register isolated environment
            self.isolated_environments[environment_id] = isolated_env
            
            logger.info(f"Successfully created isolated test environment: {environment_id}")
            return isolated_env
            
        except Exception as e:
            logger.error(f"Failed to create isolated test environment {environment_id}: {e}")
            raise
    
    def _configure_test_isolation(
        self,
        isolated_env: IsolationEnvironment,
        core_isolation_result: Dict[str, Any]
    ) -> None:
        """Configure test-specific isolation settings."""
        try:
            # Configure Python path isolation
            if isolated_env.isolation_config.enable_dependency_isolation:
                isolated_env.python_path = self._create_isolated_python_path(
                    isolated_env.test_workspace_path
                )
            
            # Configure environment variable isolation
            if isolated_env.isolation_config.enable_environment_isolation:
                isolated_env.environment_variables = self._create_isolated_environment_variables(
                    isolated_env.test_workspace_path
                )
            
            # Configure resource limits
            if isolated_env.isolation_config.enable_resource_isolation:
                isolated_env.resource_limits = self._create_resource_limits(
                    isolated_env.isolation_config
                )
            
            # Update with core isolation results
            isolated_env.resource_limits.update(
                core_isolation_result.get("resource_limits", {})
            )
            
            logger.debug(f"Configured test isolation for: {isolated_env.environment_id}")
            
        except Exception as e:
            logger.warning(f"Failed to configure test isolation: {e}")
    
    def _create_isolated_python_path(self, test_workspace_path: str) -> List[str]:
        """Create isolated Python path for test environment."""
        python_path = []
        
        try:
            workspace_path = Path(test_workspace_path)
            
            # Add test workspace to Python path
            src_path = workspace_path / "src"
            if src_path.exists():
                python_path.append(str(src_path))
            
            # Add cursus_dev path
            cursus_dev_path = workspace_path / "src" / "cursus_dev"
            if cursus_dev_path.exists():
                python_path.append(str(cursus_dev_path))
            
            # Add system Python paths (filtered)
            system_paths = sys.path.copy()
            for path in system_paths:
                path_obj = Path(path)
                # Only include allowed system paths
                if any(str(path_obj).startswith(allowed) for allowed in self.isolation_config.allowed_system_paths):
                    if path not in python_path:
                        python_path.append(path)
            
            logger.debug(f"Created isolated Python path with {len(python_path)} entries")
            return python_path
            
        except Exception as e:
            logger.warning(f"Failed to create isolated Python path: {e}")
            return sys.path.copy()
    
    def _create_isolated_environment_variables(self, test_workspace_path: str) -> Dict[str, str]:
        """Create isolated environment variables for test environment."""
        env_vars = {}
        
        try:
            workspace_path = Path(test_workspace_path)
            
            # Set test-specific environment variables
            env_vars.update({
                "CURSUS_TEST_WORKSPACE": str(workspace_path),
                "CURSUS_TEST_MODE": "isolated",
                "PYTHONPATH": ":".join(self._create_isolated_python_path(test_workspace_path)),
                "CURSUS_WORKSPACE_ROOT": str(workspace_path.parent),
                "CURSUS_ISOLATION_ENABLED": "true"
            })
            
            # Copy safe system environment variables
            safe_env_vars = [
                "PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
                "TMPDIR", "TMP", "TEMP"
            ]
            
            for var in safe_env_vars:
                if var in os.environ:
                    env_vars[var] = os.environ[var]
            
            logger.debug(f"Created isolated environment with {len(env_vars)} variables")
            return env_vars
            
        except Exception as e:
            logger.warning(f"Failed to create isolated environment variables: {e}")
            return os.environ.copy()
    
    def _create_resource_limits(self, config: WorkspaceIsolationConfig) -> Dict[str, Any]:
        """Create resource limits for test environment."""
        limits = {}
        
        if config.max_memory_mb:
            limits["max_memory_bytes"] = config.max_memory_mb * 1024 * 1024
        
        if config.max_cpu_percent:
            limits["max_cpu_percent"] = config.max_cpu_percent
        
        if config.timeout_seconds:
            limits["timeout_seconds"] = config.timeout_seconds
        
        return limits
    
    # Test Isolation Validation
    
    def validate_test_isolation(
        self,
        environment_id: str,
        strict: bool = False
    ) -> Tuple[bool, List[IsolationViolation]]:
        """
        Validate test environment isolation.
        
        Args:
            environment_id: Environment identifier to validate
            strict: Whether to apply strict validation rules
        
        Returns:
            Tuple of (is_isolated, list_of_violations)
        """
        logger.info(f"Validating test isolation for environment: {environment_id}")
        
        if environment_id not in self.isolated_environments:
            violation = IsolationViolation(
                violation_type="environment",
                severity="critical",
                description=f"Environment not found: {environment_id}",
                recommendation="Create the test environment before validation"
            )
            return False, [violation]
        
        isolated_env = self.isolated_environments[environment_id]
        violations = []
        
        try:
            # Use Phase 1 isolation manager for core validation
            core_validation_result = self.core_isolation_manager.validate_workspace_boundaries(
                workspace_path=isolated_env.test_workspace_path
            )
            
            # Convert core validation issues to isolation violations
            if not core_validation_result.is_valid:
                for issue in core_validation_result.issues:
                    violations.append(IsolationViolation(
                        violation_type="workspace_boundary",
                        severity="high",
                        description=f"Core isolation violation: {issue}",
                        recommendation="Fix workspace boundary configuration"
                    ))
            
            # Test-specific isolation validation
            violations.extend(self._validate_path_isolation(isolated_env, strict))
            violations.extend(self._validate_environment_isolation(isolated_env, strict))
            violations.extend(self._validate_dependency_isolation(isolated_env, strict))
            violations.extend(self._validate_resource_isolation(isolated_env, strict))
            
            is_isolated = len(violations) == 0
            
            logger.info(f"Test isolation validation completed: {environment_id} - "
                       f"Isolated: {is_isolated}, Violations: {len(violations)}")
            
            return is_isolated, violations
            
        except Exception as e:
            logger.error(f"Failed to validate test isolation for {environment_id}: {e}")
            violation = IsolationViolation(
                violation_type="validation_error",
                severity="critical",
                description=f"Validation error: {e}",
                recommendation="Fix validation errors and retry"
            )
            return False, [violation]
    
    def _validate_path_isolation(
        self,
        isolated_env: IsolationEnvironment,
        strict: bool
    ) -> List[IsolationViolation]:
        """Validate path isolation for test environment."""
        violations = []
        
        if not isolated_env.isolation_config.enable_path_isolation:
            return violations
        
        try:
            workspace_path = Path(isolated_env.test_workspace_path)
            
            # Check for blocked paths
            for blocked_path in isolated_env.isolation_config.blocked_paths:
                blocked_path_obj = Path(blocked_path)
                if blocked_path_obj.exists():
                    # Check if test workspace has access to blocked path
                    if self._has_path_access(workspace_path, blocked_path_obj):
                        violations.append(IsolationViolation(
                            violation_type="path",
                            severity="high",
                            description=f"Access to blocked path detected: {blocked_path}",
                            detected_path=str(blocked_path_obj),
                            recommendation=f"Remove access to blocked path: {blocked_path}"
                        ))
            
            # Check for unauthorized system access
            if strict:
                violations.extend(self._check_system_path_access(workspace_path))
            
        except Exception as e:
            violations.append(IsolationViolation(
                violation_type="path",
                severity="medium",
                description=f"Path isolation validation error: {e}",
                recommendation="Review path isolation configuration"
            ))
        
        return violations
    
    def _validate_environment_isolation(
        self,
        isolated_env: IsolationEnvironment,
        strict: bool
    ) -> List[IsolationViolation]:
        """Validate environment variable isolation."""
        violations = []
        
        if not isolated_env.isolation_config.enable_environment_isolation:
            return violations
        
        try:
            # Check for dangerous environment variables
            dangerous_vars = [
                "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                "PYTHONPATH"  # Should be controlled by isolation
            ]
            
            current_env = os.environ
            for var in dangerous_vars:
                if var in current_env and var not in isolated_env.environment_variables:
                    violations.append(IsolationViolation(
                        violation_type="environment",
                        severity="medium",
                        description=f"Dangerous environment variable not isolated: {var}",
                        recommendation=f"Ensure {var} is properly isolated or controlled"
                    ))
            
            # Check PYTHONPATH isolation
            if "PYTHONPATH" in isolated_env.environment_variables:
                pythonpath = isolated_env.environment_variables["PYTHONPATH"]
                if not self._is_pythonpath_isolated(pythonpath, isolated_env.test_workspace_path):
                    violations.append(IsolationViolation(
                        violation_type="environment",
                        severity="high",
                        description="PYTHONPATH not properly isolated",
                        recommendation="Ensure PYTHONPATH only includes test workspace paths"
                    ))
            
        except Exception as e:
            violations.append(IsolationViolation(
                violation_type="environment",
                severity="medium",
                description=f"Environment isolation validation error: {e}",
                recommendation="Review environment isolation configuration"
            ))
        
        return violations
    
    def _validate_dependency_isolation(
        self,
        isolated_env: IsolationEnvironment,
        strict: bool
    ) -> List[IsolationViolation]:
        """Validate dependency isolation for test environment."""
        violations = []
        
        if not isolated_env.isolation_config.enable_dependency_isolation:
            return violations
        
        try:
            # Check Python path isolation
            for path in isolated_env.python_path:
                path_obj = Path(path)
                
                # Ensure paths are within allowed areas
                workspace_path = Path(isolated_env.test_workspace_path)
                is_workspace_path = str(path_obj).startswith(str(workspace_path))
                is_allowed_system_path = any(
                    str(path_obj).startswith(allowed)
                    for allowed in isolated_env.isolation_config.allowed_system_paths
                )
                
                if not (is_workspace_path or is_allowed_system_path):
                    violations.append(IsolationViolation(
                        violation_type="dependency",
                        severity="medium",
                        description=f"Unauthorized path in Python path: {path}",
                        detected_path=str(path_obj),
                        recommendation="Remove unauthorized paths from Python path"
                    ))
            
            # Check for dependency conflicts (if strict)
            if strict:
                violations.extend(self._check_dependency_conflicts(isolated_env))
            
        except Exception as e:
            violations.append(IsolationViolation(
                violation_type="dependency",
                severity="medium",
                description=f"Dependency isolation validation error: {e}",
                recommendation="Review dependency isolation configuration"
            ))
        
        return violations
    
    def _validate_resource_isolation(
        self,
        isolated_env: IsolationEnvironment,
        strict: bool
    ) -> List[IsolationViolation]:
        """Validate resource isolation for test environment."""
        violations = []
        
        if not isolated_env.isolation_config.enable_resource_isolation:
            return violations
        
        try:
            # Check resource limits are configured
            if not isolated_env.resource_limits:
                if strict:
                    violations.append(IsolationViolation(
                        violation_type="resource",
                        severity="low",
                        description="No resource limits configured",
                        recommendation="Configure resource limits for better isolation"
                    ))
            
            # Validate resource limit values
            if "max_memory_bytes" in isolated_env.resource_limits:
                max_memory = isolated_env.resource_limits["max_memory_bytes"]
                if max_memory <= 0:
                    violations.append(IsolationViolation(
                        violation_type="resource",
                        severity="medium",
                        description="Invalid memory limit configuration",
                        recommendation="Set valid memory limits"
                    ))
            
            if "max_cpu_percent" in isolated_env.resource_limits:
                max_cpu = isolated_env.resource_limits["max_cpu_percent"]
                if max_cpu <= 0 or max_cpu > 100:
                    violations.append(IsolationViolation(
                        violation_type="resource",
                        severity="medium",
                        description="Invalid CPU limit configuration",
                        recommendation="Set valid CPU limits (0-100%)"
                    ))
            
        except Exception as e:
            violations.append(IsolationViolation(
                violation_type="resource",
                severity="medium",
                description=f"Resource isolation validation error: {e}",
                recommendation="Review resource isolation configuration"
            ))
        
        return violations
    
    # Helper Methods
    
    def _has_path_access(self, workspace_path: Path, target_path: Path) -> bool:
        """Check if workspace has access to target path."""
        try:
            # Simple check - in practice, this would be more sophisticated
            # Check if target path is accessible from workspace
            return target_path.exists() and os.access(target_path, os.R_OK)
        except Exception:
            return False
    
    def _check_system_path_access(self, workspace_path: Path) -> List[IsolationViolation]:
        """Check for unauthorized system path access."""
        violations = []
        
        # Check for common system directories that should be restricted
        restricted_paths = ["/etc", "/var", "/root", "/boot"]
        
        for restricted_path in restricted_paths:
            restricted_path_obj = Path(restricted_path)
            if restricted_path_obj.exists():
                if self._has_path_access(workspace_path, restricted_path_obj):
                    violations.append(IsolationViolation(
                        violation_type="path",
                        severity="high",
                        description=f"Unauthorized access to system path: {restricted_path}",
                        detected_path=restricted_path,
                        recommendation=f"Block access to system path: {restricted_path}"
                    ))
        
        return violations
    
    def _is_pythonpath_isolated(self, pythonpath: str, workspace_path: str) -> bool:
        """Check if PYTHONPATH is properly isolated."""
        try:
            paths = pythonpath.split(":")
            workspace_path_obj = Path(workspace_path)
            
            for path in paths:
                path_obj = Path(path)
                
                # Check if path is within workspace or allowed system paths
                is_workspace_path = str(path_obj).startswith(str(workspace_path_obj))
                is_allowed_system_path = any(
                    str(path_obj).startswith(allowed)
                    for allowed in self.isolation_config.allowed_system_paths
                )
                
                if not (is_workspace_path or is_allowed_system_path):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _check_dependency_conflicts(self, isolated_env: IsolationEnvironment) -> List[IsolationViolation]:
        """Check for dependency conflicts in isolated environment."""
        violations = []
        
        try:
            # This is a simplified check - in practice, you might check for:
            # - Version conflicts
            # - Circular dependencies
            # - Missing dependencies
            # - Unauthorized dependencies
            
            # For now, just check for basic path conflicts
            seen_paths = set()
            for path in isolated_env.python_path:
                if path in seen_paths:
                    violations.append(IsolationViolation(
                        violation_type="dependency",
                        severity="low",
                        description=f"Duplicate path in Python path: {path}",
                        detected_path=path,
                        recommendation="Remove duplicate paths from Python path"
                    ))
                seen_paths.add(path)
            
        except Exception as e:
            violations.append(IsolationViolation(
                violation_type="dependency",
                severity="low",
                description=f"Dependency conflict check error: {e}",
                recommendation="Review dependency configuration"
            ))
        
        return violations
    
    # Environment Management
    
    def suspend_isolated_environment(self, environment_id: str) -> bool:
        """Suspend an isolated test environment."""
        logger.info(f"Suspending isolated environment: {environment_id}")
        
        if environment_id not in self.isolated_environments:
            logger.warning(f"Environment not found: {environment_id}")
            return False
        
        try:
            isolated_env = self.isolated_environments[environment_id]
            isolated_env.status = "suspended"
            
            logger.info(f"Successfully suspended environment: {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suspend environment {environment_id}: {e}")
            return False
    
    def terminate_isolated_environment(self, environment_id: str) -> bool:
        """Terminate an isolated test environment."""
        logger.info(f"Terminating isolated environment: {environment_id}")
        
        if environment_id not in self.isolated_environments:
            logger.warning(f"Environment not found: {environment_id}")
            return False
        
        try:
            isolated_env = self.isolated_environments[environment_id]
            isolated_env.status = "terminated"
            
            # Clean up isolation resources
            self._cleanup_isolation_resources(isolated_env)
            
            # Remove from active environments
            del self.isolated_environments[environment_id]
            
            logger.info(f"Successfully terminated environment: {environment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate environment {environment_id}: {e}")
            return False
    
    def _cleanup_isolation_resources(self, isolated_env: IsolationEnvironment) -> None:
        """Clean up resources for isolated environment."""
        try:
            # Clean up temporary files, processes, etc.
            # This is a placeholder for actual cleanup logic
            logger.debug(f"Cleaned up isolation resources for: {isolated_env.environment_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup isolation resources: {e}")
    
    # Information and Statistics
    
    def get_isolation_info(self, environment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get isolation information for environments."""
        if environment_id:
            if environment_id not in self.isolated_environments:
                return {"error": f"Environment not found: {environment_id}"}
            
            isolated_env = self.isolated_environments[environment_id]
            return {
                "environment_id": isolated_env.environment_id,
                "test_workspace_path": isolated_env.test_workspace_path,
                "status": isolated_env.status,
                "created_at": isolated_env.created_at.isoformat(),
                "isolation_config": isolated_env.isolation_config.model_dump(),
                "python_path_entries": len(isolated_env.python_path),
                "environment_variables": len(isolated_env.environment_variables),
                "resource_limits": isolated_env.resource_limits
            }
        else:
            return {
                "total_environments": len(self.isolated_environments),
                "environments": {
                    env_id: {
                        "status": env.status,
                        "created_at": env.created_at.isoformat(),
                        "workspace_path": env.test_workspace_path
                    }
                    for env_id, env in self.isolated_environments.items()
                },
                "phase1_integration": {
                    "core_isolation_manager": str(type(self.core_isolation_manager)),
                    "workspace_manager": str(type(self.workspace_manager))
                }
            }
    
    def get_isolation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive isolation statistics."""
        try:
            return {
                "isolated_environments": {
                    "total": len(self.isolated_environments),
                    "by_status": {
                        status: len([
                            env for env in self.isolated_environments.values()
                            if env.status == status
                        ])
                        for status in ["active", "suspended", "terminated"]
                    }
                },
                "isolation_features": {
                    "path_isolation_enabled": self.isolation_config.enable_path_isolation,
                    "environment_isolation_enabled": self.isolation_config.enable_environment_isolation,
                    "dependency_isolation_enabled": self.isolation_config.enable_dependency_isolation,
                    "resource_isolation_enabled": self.isolation_config.enable_resource_isolation
                },
                "phase1_integration_status": {
                    "core_isolation_integrated": True,
                    "workspace_manager_integrated": True
                }
            }
        except Exception as e:
            logger.error(f"Failed to get isolation statistics: {e}")
            return {"error": str(e)}


# Convenience functions

def create_test_isolation_manager(
    workspace_root: Optional[str] = None,
    isolation_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WorkspaceTestIsolationManager:
    """
    Convenience function to create a configured WorkspaceTestIsolationManager.
    
    Args:
        workspace_root: Root directory for workspaces
        isolation_config: Test isolation configuration
        **kwargs: Additional arguments for WorkspaceManager
    
    Returns:
        Configured WorkspaceTestIsolationManager instance
    """
    # Create Phase 1 consolidated workspace manager
    workspace_manager = WorkspaceManager(workspace_root=workspace_root, **kwargs)
    
    # Create isolation configuration if provided
    test_isolation_config = None
    if isolation_config:
        test_isolation_config = WorkspaceIsolationConfig(**isolation_config)
    
    return WorkspaceTestIsolationManager(
        workspace_manager=workspace_manager,
        isolation_config=test_isolation_config
    )


def validate_test_environment_isolation(
    test_workspace_path: str,
    strict: bool = False,
    isolation_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Convenience function to validate test environment isolation.
    
    Args:
        test_workspace_path: Path to test workspace
        strict: Whether to apply strict validation rules
        isolation_config: Optional isolation configuration
    
    Returns:
        Tuple of (is_isolated, list_of_violation_dicts)
    """
    isolation_manager = create_test_isolation_manager(
        workspace_root=str(Path(test_workspace_path).parent),
        isolation_config=isolation_config
    )
    
    # Create isolated environment
    isolated_env = isolation_manager.create_isolated_test_environment(
        test_workspace_path=test_workspace_path
    )
    
    # Validate isolation
    is_isolated, violations = isolation_manager.validate_test_isolation(
        isolated_env.environment_id,
        strict=strict
    )
    
    # Convert violations to dictionaries
    violation_dicts = [violation.model_dump() for violation in violations]
    
    return is_isolated, violation_dicts
