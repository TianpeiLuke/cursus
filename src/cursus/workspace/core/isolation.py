"""
Workspace Isolation Manager

Manages workspace isolation and sandboxing utilities to ensure proper
workspace boundaries and prevent cross-workspace interference.

Features:
- Workspace boundary validation and enforcement
- Path isolation and access control
- Namespace isolation management
- Isolation violation detection and reporting
- Workspace health monitoring and validation
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class IsolationViolation(BaseModel):
    """Represents a workspace isolation violation."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    violation_type: str  # "path_access", "namespace_conflict", "environment", "dependency", "resource"
    workspace_id: str
    description: str
    severity: str = "medium"  # "low", "medium", "high", "critical"
    details: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.now)
    detected_path: Optional[str] = None
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary representation."""
        return self.model_dump()


class WorkspaceIsolationManager:
    """
    Workspace isolation utilities.
    
    Provides utilities for maintaining workspace isolation, validating
    workspace boundaries, and detecting isolation violations.
    """
    
    def __init__(self, workspace_manager):
        """
        Initialize workspace isolation manager.
        
        Args:
            workspace_manager: Parent WorkspaceManager instance
        """
        self.workspace_manager = workspace_manager
        self.isolation_violations: List[IsolationViolation] = []
        
        logger.info("Initialized workspace isolation manager")
    
    def validate_workspace_boundaries(self, workspace_id: str) -> Dict[str, Any]:
        """
        Validate workspace boundaries and isolation.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            Validation result dictionary
        """
        logger.info(f"Validating workspace boundaries for: {workspace_id}")
        
        validation_result = {
            "workspace_id": workspace_id,
            "valid": True,
            "violations": [],
            "warnings": [],
            "checks_performed": []
        }
        
        try:
            if workspace_id not in self.workspace_manager.active_workspaces:
                validation_result["valid"] = False
                validation_result["violations"].append("Workspace not found in active workspaces")
                return validation_result
            
            workspace_context = self.workspace_manager.active_workspaces[workspace_id]
            workspace_path = Path(workspace_context.workspace_path)
            
            # Check 1: Workspace path isolation
            path_check = self._validate_path_isolation(workspace_id, workspace_path)
            validation_result["checks_performed"].append("path_isolation")
            if not path_check["valid"]:
                validation_result["valid"] = False
                validation_result["violations"].extend(path_check["violations"])
            validation_result["warnings"].extend(path_check.get("warnings", []))
            
            # Check 2: Namespace isolation
            namespace_check = self._validate_namespace_isolation(workspace_id, workspace_path)
            validation_result["checks_performed"].append("namespace_isolation")
            if not namespace_check["valid"]:
                validation_result["valid"] = False
                validation_result["violations"].extend(namespace_check["violations"])
            validation_result["warnings"].extend(namespace_check.get("warnings", []))
            
            # Check 3: Import isolation
            import_check = self._validate_import_isolation(workspace_id, workspace_path)
            validation_result["checks_performed"].append("import_isolation")
            if not import_check["valid"]:
                validation_result["valid"] = False
                validation_result["violations"].extend(import_check["violations"])
            validation_result["warnings"].extend(import_check.get("warnings", []))
            
            logger.info(f"Workspace boundary validation completed for {workspace_id}: {'PASS' if validation_result['valid'] else 'FAIL'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate workspace boundaries for {workspace_id}: {e}")
            validation_result["valid"] = False
            validation_result["violations"].append(f"Validation error: {e}")
            return validation_result
    
    def _validate_path_isolation(self, workspace_id: str, workspace_path: Path) -> Dict[str, Any]:
        """Validate path isolation for workspace."""
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Check that workspace path is within expected boundaries
            workspace_root = self.workspace_manager.workspace_root
            if not workspace_path.is_relative_to(workspace_root):
                result["valid"] = False
                result["violations"].append(f"Workspace path outside workspace root: {workspace_path}")
            
            # Check for symlinks that could break isolation
            for item in workspace_path.rglob("*"):
                if item.is_symlink():
                    target = item.resolve()
                    if not target.is_relative_to(workspace_root):
                        result["valid"] = False
                        result["violations"].append(f"Symlink breaks isolation: {item} -> {target}")
            
            # Check for absolute paths in configuration files
            config_files = list(workspace_path.glob("**/*.json")) + list(workspace_path.glob("**/*.yaml"))
            for config_file in config_files:
                if self._contains_absolute_paths_outside_workspace(config_file, workspace_root):
                    result["warnings"].append(f"Configuration file may contain absolute paths: {config_file}")
            
        except Exception as e:
            result["valid"] = False
            result["violations"].append(f"Path isolation validation error: {e}")
        
        return result
    
    def _validate_namespace_isolation(self, workspace_id: str, workspace_path: Path) -> Dict[str, Any]:
        """Validate namespace isolation for workspace."""
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Check for namespace conflicts in Python modules
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            if cursus_dev_dir.exists():
                # Check for module name conflicts
                module_names = set()
                for module_dir in cursus_dev_dir.iterdir():
                    if module_dir.is_dir() and module_dir.name != "__pycache__":
                        for py_file in module_dir.glob("*.py"):
                            if py_file.name != "__init__.py":
                                module_name = py_file.stem
                                if module_name in module_names:
                                    result["warnings"].append(f"Duplicate module name: {module_name}")
                                module_names.add(module_name)
            
            # Check for global variable conflicts
            global_vars = self._detect_global_variable_usage(workspace_path)
            if global_vars:
                result["warnings"].extend([
                    f"Global variable usage detected: {var}" for var in global_vars
                ])
            
        except Exception as e:
            result["valid"] = False
            result["violations"].append(f"Namespace isolation validation error: {e}")
        
        return result
    
    def _validate_import_isolation(self, workspace_id: str, workspace_path: Path) -> Dict[str, Any]:
        """Validate import isolation for workspace."""
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Check for imports that break workspace isolation
            python_files = list(workspace_path.glob("**/*.py"))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for absolute imports outside workspace
                    problematic_imports = self._detect_problematic_imports(content, workspace_path)
                    if problematic_imports:
                        result["warnings"].extend([
                            f"Potentially problematic import in {py_file}: {imp}"
                            for imp in problematic_imports
                        ])
                
                except Exception as e:
                    result["warnings"].append(f"Could not analyze imports in {py_file}: {e}")
            
        except Exception as e:
            result["valid"] = False
            result["violations"].append(f"Import isolation validation error: {e}")
        
        return result
    
    def _contains_absolute_paths_outside_workspace(self, config_file: Path, workspace_root: Path) -> bool:
        """Check if configuration file contains absolute paths outside workspace."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for absolute path patterns
            import re
            absolute_path_pattern = r'["\'](/[^"\']*)["\']'
            matches = re.findall(absolute_path_pattern, content)
            
            for match in matches:
                abs_path = Path(match)
                if abs_path.exists() and not abs_path.is_relative_to(workspace_root):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_global_variable_usage(self, workspace_path: Path) -> List[str]:
        """Detect global variable usage in workspace."""
        global_vars = []
        
        try:
            python_files = list(workspace_path.glob("**/*.py"))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple detection of global keyword usage
                    if "global " in content:
                        import re
                        global_matches = re.findall(r'global\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                        global_vars.extend([f"{py_file.name}:{var}" for var in global_matches])
                
                except Exception:
                    continue
            
        except Exception:
            pass
        
        return global_vars
    
    def _detect_problematic_imports(self, content: str, workspace_path: Path) -> List[str]:
        """Detect potentially problematic imports."""
        problematic = []
        
        try:
            import re
            
            # Look for imports that might break isolation
            import_patterns = [
                r'from\s+(?!cursus|cursus_dev)([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'import\s+(?!cursus|cursus_dev)([a-zA-Z_][a-zA-Z0-9_.]*)'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Filter out standard library and common packages
                    if not self._is_safe_import(match):
                        problematic.append(match)
        
        except Exception:
            pass
        
        return problematic
    
    def _is_safe_import(self, import_name: str) -> bool:
        """Check if import is considered safe for workspace isolation."""
        safe_imports = {
            # Standard library modules
            "os", "sys", "json", "yaml", "pathlib", "typing", "logging",
            "datetime", "time", "re", "collections", "itertools", "functools",
            "math", "random", "uuid", "hashlib", "base64", "urllib", "http",
            
            # Common data science packages
            "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
            "plotly", "bokeh", "jupyter", "ipython",
            
            # AWS and SageMaker packages
            "boto3", "botocore", "sagemaker", "awscli",
            
            # Common utility packages
            "pydantic", "click", "requests", "pytest", "mock"
        }
        
        # Check if import starts with any safe import
        return any(import_name.startswith(safe) for safe in safe_imports)
    
    def enforce_path_isolation(
        self,
        workspace_path: str,
        access_path: str
    ) -> bool:
        """
        Enforce path isolation for workspace access.
        
        Args:
            workspace_path: Workspace root path
            access_path: Path being accessed
        
        Returns:
            True if access is allowed, False otherwise
        """
        try:
            workspace_path = Path(workspace_path).resolve()
            access_path = Path(access_path).resolve()
            
            # Check if access path is within workspace boundaries
            if access_path.is_relative_to(workspace_path):
                return True
            
            # Check if access path is in shared workspace (if enabled)
            if self.workspace_manager.config and self.workspace_manager.config.enable_shared_fallback:
                shared_path = self.workspace_manager.workspace_root / "shared"
                if access_path.is_relative_to(shared_path):
                    return True
            
            # Check if access path is in core cursus package (always allowed)
            if "src/cursus" in str(access_path):
                return True
            
            # Access denied
            logger.warning(f"Path isolation violation: {access_path} not accessible from {workspace_path}")
            return False
            
        except Exception as e:
            logger.error(f"Error enforcing path isolation: {e}")
            return False
    
    def manage_namespace_isolation(
        self,
        workspace_id: str,
        component_name: str
    ) -> str:
        """
        Manage namespace isolation for workspace components.
        
        Args:
            workspace_id: Workspace identifier
            component_name: Component name to namespace
        
        Returns:
            Namespaced component name
        """
        try:
            # Create namespaced component name
            if workspace_id == "shared":
                return component_name
            else:
                return f"{workspace_id}:{component_name}"
                
        except Exception as e:
            logger.error(f"Error managing namespace isolation: {e}")
            return component_name
    
    def detect_isolation_violations(self, workspace_id: str) -> List[IsolationViolation]:
        """
        Detect isolation violations in workspace.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            List of detected isolation violations
        """
        logger.info(f"Detecting isolation violations for: {workspace_id}")
        
        violations = []
        
        try:
            if workspace_id not in self.workspace_manager.active_workspaces:
                violation = IsolationViolation(
                    violation_type="workspace_not_found",
                    workspace_id=workspace_id,
                    description="Workspace not found in active workspaces",
                    severity="high"
                )
                violations.append(violation)
                return violations
            
            workspace_context = self.workspace_manager.active_workspaces[workspace_id]
            workspace_path = Path(workspace_context.workspace_path)
            
            # Detect various types of violations
            violations.extend(self._detect_path_violations(workspace_id, workspace_path))
            violations.extend(self._detect_namespace_violations(workspace_id, workspace_path))
            violations.extend(self._detect_import_violations(workspace_id, workspace_path))
            
            # Store violations for tracking
            self.isolation_violations.extend(violations)
            
            logger.info(f"Detected {len(violations)} isolation violations for {workspace_id}")
            return violations
            
        except Exception as e:
            logger.error(f"Failed to detect isolation violations for {workspace_id}: {e}")
            violation = IsolationViolation(
                violation_type="detection_error",
                workspace_id=workspace_id,
                description=f"Error detecting violations: {e}",
                severity="medium"
            )
            return [violation]
    
    def _detect_path_violations(self, workspace_id: str, workspace_path: Path) -> List[IsolationViolation]:
        """Detect path-related isolation violations."""
        violations = []
        
        try:
            # Check for symlinks outside workspace
            for item in workspace_path.rglob("*"):
                if item.is_symlink():
                    target = item.resolve()
                    if not target.is_relative_to(self.workspace_manager.workspace_root):
                        violation = IsolationViolation(
                            violation_type="external_symlink",
                            workspace_id=workspace_id,
                            description=f"Symlink points outside workspace: {item} -> {target}",
                            severity="high",
                            details={"symlink_path": str(item), "target_path": str(target)}
                        )
                        violations.append(violation)
        
        except Exception as e:
            logger.error(f"Error detecting path violations: {e}")
        
        return violations
    
    def _detect_namespace_violations(self, workspace_id: str, workspace_path: Path) -> List[IsolationViolation]:
        """Detect namespace-related isolation violations."""
        violations = []
        
        try:
            # Check for module name conflicts with core system
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            if cursus_dev_dir.exists():
                core_module_names = self._get_core_module_names()
                
                for module_dir in cursus_dev_dir.iterdir():
                    if module_dir.is_dir() and module_dir.name != "__pycache__":
                        for py_file in module_dir.glob("*.py"):
                            if py_file.name != "__init__.py":
                                module_name = py_file.stem
                                if module_name in core_module_names:
                                    violation = IsolationViolation(
                                        violation_type="namespace_conflict",
                                        workspace_id=workspace_id,
                                        description=f"Module name conflicts with core system: {module_name}",
                                        severity="medium",
                                        details={"module_file": str(py_file), "conflicting_name": module_name}
                                    )
                                    violations.append(violation)
        
        except Exception as e:
            logger.error(f"Error detecting namespace violations: {e}")
        
        return violations
    
    def _detect_import_violations(self, workspace_id: str, workspace_path: Path) -> List[IsolationViolation]:
        """Detect import-related isolation violations."""
        violations = []
        
        try:
            python_files = list(workspace_path.glob("**/*.py"))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for direct imports of other workspace modules
                    cross_workspace_imports = self._detect_cross_workspace_imports(content)
                    for imp in cross_workspace_imports:
                        violation = IsolationViolation(
                            violation_type="cross_workspace_import",
                            workspace_id=workspace_id,
                            description=f"Direct import of other workspace module: {imp}",
                            severity="high",
                            details={"file": str(py_file), "import": imp}
                        )
                        violations.append(violation)
                
                except Exception as e:
                    logger.warning(f"Could not analyze imports in {py_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error detecting import violations: {e}")
        
        return violations
    
    def _get_core_module_names(self) -> set:
        """Get set of core module names to check for conflicts."""
        core_names = set()
        
        try:
            # Get core cursus module names
            cursus_src = Path(__file__).parent.parent.parent
            for py_file in cursus_src.rglob("*.py"):
                if py_file.name != "__init__.py":
                    core_names.add(py_file.stem)
        
        except Exception as e:
            logger.warning(f"Could not get core module names: {e}")
        
        return core_names
    
    def _detect_cross_workspace_imports(self, content: str) -> List[str]:
        """Detect imports that reference other workspaces directly."""
        cross_workspace_imports = []
        
        try:
            import re
            
            # Look for imports that reference other developer workspaces
            patterns = [
                r'from\s+development\.developers\.([^.]+)',
                r'import\s+development\.developers\.([^.]+)',
                r'from\s+cursus_dev\.([^.]+)\.steps',
                r'import\s+cursus_dev\.([^.]+)\.steps'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                cross_workspace_imports.extend(matches)
        
        except Exception:
            pass
        
        return cross_workspace_imports
    
    def validate_workspace_structure(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate workspace structure.
        
        Args:
            workspace_root: Root directory to validate
            strict: Whether to apply strict validation rules
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if workspace_root:
            workspace_root = Path(workspace_root)
        else:
            workspace_root = self.workspace_manager.workspace_root
        
        if not workspace_root:
            return False, ["No workspace root specified"]
        
        logger.info(f"Validating workspace structure: {workspace_root}")
        
        issues = []
        
        try:
            # Check workspace root exists
            if not workspace_root.exists():
                issues.append(f"Workspace root does not exist: {workspace_root}")
                return False, issues
            
            # Check for required directories
            developers_dir = workspace_root / "developers"
            shared_dir = workspace_root / "shared"
            
            if not developers_dir.exists() and not shared_dir.exists():
                issues.append("Workspace must contain 'developers' or 'shared' directory")
            
            # Validate developer workspaces
            if developers_dir.exists():
                dev_issues = self._validate_development(developers_dir, strict)
                issues.extend(dev_issues)
            
            # Validate shared workspace
            if shared_dir.exists():
                shared_issues = self._validate_shared_workspace(shared_dir, strict)
                issues.extend(shared_issues)
            
            is_valid = len(issues) == 0
            logger.info(f"Workspace structure validation: {'PASS' if is_valid else 'FAIL'} ({len(issues)} issues)")
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Failed to validate workspace structure: {e}")
            return False, [f"Validation error: {e}"]
    
    def _validate_development(
        self,
        developers_dir: Path,
        strict: bool
    ) -> List[str]:
        """Validate developer workspace structures."""
        issues = []
        
        try:
            if not any(developers_dir.iterdir()):
                if strict:
                    issues.append("No developer workspaces found")
                return issues
            
            for item in developers_dir.iterdir():
                if not item.is_dir():
                    continue
                
                developer_id = item.name
                cursus_dev_dir = item / "src" / "cursus_dev" / "steps"
                
                if not cursus_dev_dir.exists():
                    issues.append(f"Developer '{developer_id}' missing cursus_dev structure")
                    continue
                
                # Check for at least one module type directory
                module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
                has_any_module_dir = any(
                    (cursus_dev_dir / module_dir).exists()
                    for module_dir in module_dirs
                )
                
                if strict and not has_any_module_dir:
                    issues.append(f"Developer '{developer_id}' has no module directories")
        
        except Exception as e:
            issues.append(f"Error validating developer workspaces: {e}")
        
        return issues
    
    def _validate_shared_workspace(
        self,
        shared_dir: Path,
        strict: bool
    ) -> List[str]:
        """Validate shared workspace structure."""
        issues = []
        
        try:
            cursus_dev_dir = shared_dir / "src" / "cursus_dev" / "steps"
            
            if not cursus_dev_dir.exists():
                issues.append("Shared workspace missing cursus_dev structure")
                return issues
            
            # Check for at least one module type directory
            if strict:
                module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
                has_any_module_dir = any(
                    (cursus_dev_dir / module_dir).exists()
                    for module_dir in module_dirs
                )
                
                if not has_any_module_dir:
                    issues.append("Shared workspace has no module directories")
        
        except Exception as e:
            issues.append(f"Error validating shared workspace: {e}")
        
        return issues
    
    def get_workspace_health(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get health information for a workspace.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            Health information dictionary
        """
        logger.info(f"Getting workspace health for: {workspace_id}")
        
        health_info = {
            "workspace_id": workspace_id,
            "healthy": True,
            "issues": [],
            "warnings": [],
            "last_checked": datetime.now().isoformat(),
            "health_score": 100
        }
        
        try:
            if workspace_id not in self.workspace_manager.active_workspaces:
                health_info["healthy"] = False
                health_info["issues"].append("Workspace not found")
                health_info["health_score"] = 0
                return health_info
            
            workspace_context = self.workspace_manager.active_workspaces[workspace_id]
            workspace_path = Path(workspace_context.workspace_path)
            
            # Check workspace accessibility
            if not workspace_path.exists():
                health_info["healthy"] = False
                health_info["issues"].append("Workspace path does not exist")
                health_info["health_score"] -= 50
            
            # Check isolation boundaries
            boundary_validation = self.validate_workspace_boundaries(workspace_id)
            if not boundary_validation["valid"]:
                health_info["healthy"] = False
                health_info["issues"].extend(boundary_validation["violations"])
                health_info["health_score"] -= 30
            
            health_info["warnings"].extend(boundary_validation.get("warnings", []))
            
            # Check workspace structure
            structure_valid, structure_issues = self.validate_workspace_structure(workspace_path)
            if not structure_valid:
                health_info["issues"].extend(structure_issues)
                health_info["health_score"] -= 20
            
            # Adjust health score based on warnings
            health_info["health_score"] -= min(len(health_info["warnings"]) * 5, 20)
            health_info["health_score"] = max(health_info["health_score"], 0)
            
            # Determine overall health
            health_info["healthy"] = health_info["health_score"] >= 70
            
            return health_info
            
        except Exception as e:
            logger.error(f"Failed to get workspace health for {workspace_id}: {e}")
            health_info["healthy"] = False
            health_info["issues"].append(f"Health check error: {e}")
            health_info["health_score"] = 0
            return health_info
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities."""
        try:
            return {
                "total_violations": len(self.isolation_violations),
                "violation_types": {
                    violation_type: len([
                        v for v in self.isolation_violations
                        if v.violation_type == violation_type
                    ])
                    for violation_type in set(v.violation_type for v in self.isolation_violations)
                },
                "severity_distribution": {
                    severity: len([
                        v for v in self.isolation_violations
                        if v.severity == severity
                    ])
                    for severity in ["low", "medium", "high", "critical"]
                },
                "recent_violations": [
                    v.to_dict() for v in self.isolation_violations[-5:]
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get validation summary: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get isolation management statistics."""
        try:
            return {
                "isolation_checks": {
                    "total_violations": len(self.isolation_violations),
                    "active_workspaces": len(self.workspace_manager.active_workspaces),
                    "healthy_workspaces": len([
                        ws_id for ws_id in self.workspace_manager.active_workspaces
                        if self.get_workspace_health(ws_id)["healthy"]
                    ])
                },
                "violation_summary": self.get_validation_summary()
            }
        except Exception as e:
            logger.error(f"Failed to get isolation statistics: {e}")
            return {"error": str(e)}
