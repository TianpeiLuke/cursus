"""
Workspace Unified Alignment Tester

Extends UnifiedAlignmentTester to support multi-developer workspace structures.
Provides workspace-aware alignment validation across all 4 levels while maintaining
full API compatibility with the existing UnifiedAlignmentTester.

Architecture:
- Extends existing UnifiedAlignmentTester capabilities
- Supports workspace-specific component discovery and validation
- Maintains backward compatibility with existing validation workflows
- Provides workspace-specific reporting and diagnostics

Workspace Integration:
- Uses DeveloperWorkspaceFileResolver for component discovery
- Integrates with WorkspaceModuleLoader for dynamic loading
- Supports cross-workspace dependency validation
- Provides workspace-specific error reporting and diagnostics
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import logging

from ...validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_module_loader import WorkspaceModuleLoader
from .workspace_manager import WorkspaceManager


logger = logging.getLogger(__name__)


class WorkspaceUnifiedAlignmentTester(UnifiedAlignmentTester):
    """
    Workspace-aware version of UnifiedAlignmentTester.
    
    Extends the existing UnifiedAlignmentTester to support multi-developer
    workspace structures while maintaining full API compatibility.
    
    Features:
    - Workspace-aware component discovery and validation
    - Cross-workspace dependency validation
    - Workspace-specific error reporting and diagnostics
    - Full backward compatibility with UnifiedAlignmentTester
    - Support for shared workspace fallback behavior
    """
    
    def __init__(
        self,
        workspace_root: Union[str, Path],
        developer_id: str,
        enable_shared_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize workspace-aware alignment tester.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            developer_id: Specific developer workspace to target
            enable_shared_fallback: Whether to fallback to shared workspace
            **kwargs: Additional arguments passed to UnifiedAlignmentTester
        """
        self.workspace_root = Path(workspace_root)
        self.developer_id = developer_id
        self.enable_shared_fallback = enable_shared_fallback
        
        # Initialize workspace infrastructure
        self.workspace_manager = WorkspaceManager(workspace_root=workspace_root)
        self.file_resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id=developer_id,
            enable_shared_fallback=enable_shared_fallback
        )
        self.module_loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id=developer_id,
            enable_shared_fallback=enable_shared_fallback
        )
        
        # Configure workspace-specific paths for parent class
        workspace_info = self.workspace_manager.get_workspace_info()
        if workspace_info.developers and developer_id in workspace_info.developers:
            dev_workspace = workspace_info.developers[developer_id]
            scripts_dir = dev_workspace.workspace_path / "scripts"
            contracts_dir = dev_workspace.workspace_path / "contracts"
            specs_dir = dev_workspace.workspace_path / "specs"
            builders_dir = dev_workspace.workspace_path / "builders"
            configs_dir = dev_workspace.workspace_path / "configs"
            
            # Initialize parent with workspace-relative paths
            super().__init__(
                scripts_dir=str(scripts_dir),
                contracts_dir=str(contracts_dir),
                specs_dir=str(specs_dir),
                builders_dir=str(builders_dir),
                configs_dir=str(configs_dir),
                **kwargs
            )
        else:
            # Fallback to shared workspace using src/cursus/steps as per design
            if self.enable_shared_fallback:
                # Use src/cursus/steps as the shared workspace
                shared_steps_path = self.workspace_root / "src" / "cursus" / "steps"
                if shared_steps_path.exists():
                    scripts_dir = shared_steps_path / "scripts"
                    contracts_dir = shared_steps_path / "contracts"
                    specs_dir = shared_steps_path / "specs"
                    builders_dir = shared_steps_path / "builders"
                    configs_dir = shared_steps_path / "configs"
                    
                    super().__init__(
                        scripts_dir=str(scripts_dir),
                        contracts_dir=str(contracts_dir),
                        specs_dir=str(specs_dir),
                        builders_dir=str(builders_dir),
                        configs_dir=str(configs_dir),
                        **kwargs
                    )
                else:
                    # Last resort: use default paths
                    super().__init__(**kwargs)
            else:
                # Last resort: use default paths
                super().__init__(**kwargs)
        
        logger.info(f"Initialized workspace alignment tester for developer '{developer_id}' "
                   f"at '{workspace_root}'")
    
    def run_workspace_validation(
        self,
        target_scripts: Optional[List[str]] = None,
        skip_levels: Optional[List[int]] = None,
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run alignment validation for workspace components.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            skip_levels: Validation levels to skip (None for all levels)
            workspace_context: Additional workspace context for validation
            
        Returns:
            Comprehensive validation results with workspace context
        """
        logger.info(f"Starting workspace validation for developer '{self.developer_id}'")
        
        try:
            # Discover workspace scripts if not specified
            if target_scripts is None:
                target_scripts = self._discover_workspace_scripts()
            
            # Run standard validation with workspace context
            validation_report = self.run_full_validation(
                target_scripts=target_scripts,
                skip_levels=skip_levels
            )
            
            # Convert AlignmentReport to dictionary format
            validation_results = {
                'success': validation_report.is_passing(),
                'results': {},
                'summary': validation_report.summary.model_dump() if validation_report.summary else {}
            }
            
            # Extract results from each level
            for script_name, result in validation_report.level1_results.items():
                if script_name not in validation_results['results']:
                    validation_results['results'][script_name] = {}
                validation_results['results'][script_name]['level1'] = {
                    'passed': result.passed,
                    'details': result.details
                }
            
            for script_name, result in validation_report.level2_results.items():
                if script_name not in validation_results['results']:
                    validation_results['results'][script_name] = {}
                validation_results['results'][script_name]['level2'] = {
                    'passed': result.passed,
                    'details': result.details
                }
            
            for script_name, result in validation_report.level3_results.items():
                if script_name not in validation_results['results']:
                    validation_results['results'][script_name] = {}
                validation_results['results'][script_name]['level3'] = {
                    'passed': result.passed,
                    'details': result.details
                }
            
            for script_name, result in validation_report.level4_results.items():
                if script_name not in validation_results['results']:
                    validation_results['results'][script_name] = {}
                validation_results['results'][script_name]['level4'] = {
                    'passed': result.passed,
                    'details': result.details
                }
            
            # Enhance results with workspace-specific information
            workspace_results = self._enhance_results_with_workspace_context(
                validation_results, workspace_context
            )
            
            # Add cross-workspace dependency validation if enabled
            if self.enable_shared_fallback:
                cross_workspace_results = self._validate_cross_workspace_dependencies(
                    target_scripts
                )
                workspace_results['cross_workspace_validation'] = cross_workspace_results
            
            logger.info(f"Completed workspace validation for developer '{self.developer_id}'")
            return workspace_results
            
        except Exception as e:
            logger.error(f"Workspace validation failed for developer '{self.developer_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'workspace_context': {
                    'developer_id': self.developer_id,
                    'workspace_root': str(self.workspace_root),
                    'enable_shared_fallback': self.enable_shared_fallback
                }
            }
    
    def _discover_workspace_scripts(self) -> List[str]:
        """Discover all scripts in the current workspace."""
        try:
            workspace_info = self.workspace_manager.get_workspace_info()
            if not workspace_info.developers or self.developer_id not in workspace_info.developers:
                logger.warning(f"Developer '{self.developer_id}' not found in workspace")
                return []
            
            dev_workspace = workspace_info.developers[self.developer_id]
            scripts_dir = dev_workspace.workspace_path / "scripts"
            
            if not scripts_dir.exists():
                logger.warning(f"Scripts directory not found: {scripts_dir}")
                return []
            
            # Discover Python scripts in workspace
            scripts = []
            for script_file in scripts_dir.glob("*.py"):
                if script_file.name.startswith("__"):
                    continue
                script_name = script_file.stem
                scripts.append(script_name)
            
            logger.info(f"Discovered {len(scripts)} scripts in workspace: {scripts}")
            return scripts
            
        except Exception as e:
            logger.error(f"Failed to discover workspace scripts: {e}")
            return []
    
    def _enhance_results_with_workspace_context(
        self,
        validation_results: Dict[str, Any],
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance validation results with workspace-specific context."""
        enhanced_results = validation_results.copy()
        
        # Add workspace metadata
        enhanced_results['workspace_metadata'] = {
            'developer_id': self.developer_id,
            'workspace_root': str(self.workspace_root),
            'enable_shared_fallback': self.enable_shared_fallback,
            'workspace_info': self.workspace_manager.get_workspace_info().model_dump()
        }
        
        # Add workspace context if provided
        if workspace_context:
            enhanced_results['workspace_context'] = workspace_context
        
        # Enhance error messages with workspace context
        if 'errors' in enhanced_results:
            enhanced_results['errors'] = self._add_workspace_context_to_errors(
                enhanced_results['errors']
            )
        
        # Add workspace-specific statistics
        enhanced_results['workspace_statistics'] = self._generate_workspace_statistics(
            validation_results
        )
        
        return enhanced_results
    
    def _add_workspace_context_to_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add workspace context to error messages."""
        enhanced_errors = []
        
        for error in errors:
            enhanced_error = error.copy()
            enhanced_error['workspace_context'] = {
                'developer_id': self.developer_id,
                'workspace_root': str(self.workspace_root)
            }
            
            # Enhance error message with workspace information
            if 'message' in enhanced_error:
                enhanced_error['message'] = (
                    f"[Workspace: {self.developer_id}] {enhanced_error['message']}"
                )
            
            enhanced_errors.append(enhanced_error)
        
        return enhanced_errors
    
    def _generate_workspace_statistics(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workspace-specific validation statistics."""
        stats = {
            'total_scripts_validated': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'workspace_components_found': {
                'contracts': 0,
                'specs': 0,
                'builders': 0,
                'configs': 0
            },
            'shared_components_used': 0 if not self.enable_shared_fallback else 0
        }
        
        # Count validation results
        if 'results' in validation_results:
            for script_name, script_results in validation_results['results'].items():
                stats['total_scripts_validated'] += 1
                if script_results.get('success', False):
                    stats['successful_validations'] += 1
                else:
                    stats['failed_validations'] += 1
        
        # Count workspace components
        try:
            workspace_info = self.workspace_manager.get_workspace_info()
            if workspace_info.developers and self.developer_id in workspace_info.developers:
                dev_workspace = workspace_info.developers[self.developer_id]
                
                for component_type in ['contracts', 'specs', 'builders', 'configs']:
                    component_dir = dev_workspace.workspace_path / component_type
                    if component_dir.exists():
                        component_files = list(component_dir.glob("*.py"))
                        stats['workspace_components_found'][component_type] = len(component_files)
        
        except Exception as e:
            logger.warning(f"Failed to generate workspace statistics: {e}")
        
        return stats
    
    def _validate_cross_workspace_dependencies(
        self,
        target_scripts: List[str]
    ) -> Dict[str, Any]:
        """Validate cross-workspace dependencies if shared fallback is enabled."""
        if not self.enable_shared_fallback:
            return {'enabled': False, 'message': 'Cross-workspace validation disabled'}
        
        cross_workspace_results = {
            'enabled': True,
            'shared_components_used': {},
            'dependency_conflicts': [],
            'recommendations': []
        }
        
        try:
            # Check which components fall back to shared workspace
            for script_name in target_scripts:
                shared_usage = self._check_shared_component_usage(script_name)
                if shared_usage:
                    cross_workspace_results['shared_components_used'][script_name] = shared_usage
            
            # Check for potential conflicts
            conflicts = self._detect_dependency_conflicts(target_scripts)
            cross_workspace_results['dependency_conflicts'] = conflicts
            
            # Generate recommendations
            recommendations = self._generate_cross_workspace_recommendations(
                cross_workspace_results
            )
            cross_workspace_results['recommendations'] = recommendations
            
        except Exception as e:
            logger.warning(f"Cross-workspace validation failed: {e}")
            cross_workspace_results['error'] = str(e)
        
        return cross_workspace_results
    
    def _check_shared_component_usage(self, script_name: str) -> Dict[str, bool]:
        """Check which components for a script come from shared workspace."""
        shared_usage = {}
        
        try:
            # Check each component type
            for component_type in ['contracts', 'specs', 'builders', 'configs']:
                method_name = f'find_{component_type.rstrip("s")}_file'
                if hasattr(self.file_resolver, method_name):
                    method = getattr(self.file_resolver, method_name)
                    component_path = method(script_name)
                    
                    if component_path and 'shared' in component_path:
                        shared_usage[component_type] = True
                    else:
                        shared_usage[component_type] = False
        
        except Exception as e:
            logger.warning(f"Failed to check shared component usage for {script_name}: {e}")
        
        return shared_usage
    
    def _detect_dependency_conflicts(self, target_scripts: List[str]) -> List[Dict[str, Any]]:
        """Detect potential conflicts in cross-workspace dependencies."""
        conflicts = []
        
        try:
            # This is a placeholder for more sophisticated conflict detection
            # In a full implementation, this would analyze component dependencies
            # and detect version conflicts, naming conflicts, etc.
            
            # For now, we'll do basic checks
            for script_name in target_scripts:
                shared_usage = self._check_shared_component_usage(script_name)
                mixed_usage = any(shared_usage.values()) and not all(shared_usage.values())
                
                if mixed_usage:
                    conflicts.append({
                        'script': script_name,
                        'type': 'mixed_workspace_usage',
                        'description': f'Script {script_name} uses both workspace and shared components',
                        'shared_components': [k for k, v in shared_usage.items() if v],
                        'workspace_components': [k for k, v in shared_usage.items() if not v]
                    })
        
        except Exception as e:
            logger.warning(f"Failed to detect dependency conflicts: {e}")
        
        return conflicts
    
    def _generate_cross_workspace_recommendations(
        self,
        cross_workspace_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on cross-workspace validation results."""
        recommendations = []
        
        try:
            # Recommend workspace-specific implementations for heavily shared components
            shared_usage = cross_workspace_results.get('shared_components_used', {})
            if shared_usage:
                heavily_shared_scripts = [
                    script for script, usage in shared_usage.items()
                    if sum(usage.values()) >= 3  # 3 or more shared components
                ]
                
                if heavily_shared_scripts:
                    recommendations.append(
                        f"Consider implementing workspace-specific versions of components "
                        f"for scripts: {', '.join(heavily_shared_scripts)}"
                    )
            
            # Recommend consistency for mixed usage
            conflicts = cross_workspace_results.get('dependency_conflicts', [])
            mixed_usage_scripts = [
                conflict['script'] for conflict in conflicts
                if conflict['type'] == 'mixed_workspace_usage'
            ]
            
            if mixed_usage_scripts:
                recommendations.append(
                    f"Consider using consistent component sources (all workspace or all shared) "
                    f"for scripts: {', '.join(mixed_usage_scripts)}"
                )
            
            # General recommendations
            if not recommendations:
                recommendations.append(
                    "Cross-workspace dependencies look good. "
                    "Consider implementing workspace-specific components for better isolation."
                )
        
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error.")
        
        return recommendations
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about current workspace configuration."""
        return {
            'developer_id': self.developer_id,
            'workspace_root': str(self.workspace_root),
            'enable_shared_fallback': self.enable_shared_fallback,
            'workspace_manager_info': self.workspace_manager.get_workspace_info().model_dump(),
            'file_resolver_info': self.file_resolver.get_workspace_info(),
            'available_developers': self.workspace_manager.list_available_developers()
        }
    
    def switch_developer(self, developer_id: str) -> None:
        """Switch to a different developer workspace."""
        if developer_id == self.developer_id:
            logger.info(f"Already using developer workspace: {developer_id}")
            return
        
        logger.info(f"Switching from developer '{self.developer_id}' to '{developer_id}'")
        
        # Validate new developer exists
        available_developers = self.workspace_manager.list_available_developers()
        if developer_id not in available_developers:
            raise ValueError(f"Developer workspace not found: {developer_id}")
        
        # Update developer ID
        self.developer_id = developer_id
        
        # Update file resolver
        self.file_resolver.switch_developer(developer_id)
        
        # Update module loader
        self.module_loader.switch_developer(developer_id)
        
        # Update scripts directory for parent class
        workspace_info = self.workspace_manager.get_workspace_info()
        if workspace_info.developers and developer_id in workspace_info.developers:
            dev_workspace = workspace_info.developers[developer_id]
            scripts_dir = dev_workspace.workspace_path / "scripts"
            self.scripts_directory = str(scripts_dir)
        
        logger.info(f"Successfully switched to developer workspace: {developer_id}")
    
    @classmethod
    def create_from_workspace_manager(
        cls,
        workspace_manager: WorkspaceManager,
        developer_id: str,
        **kwargs
    ) -> 'WorkspaceUnifiedAlignmentTester':
        """Create instance from existing WorkspaceManager."""
        return cls(
            workspace_root=workspace_manager.workspace_root,
            developer_id=developer_id,
            **kwargs
        )
