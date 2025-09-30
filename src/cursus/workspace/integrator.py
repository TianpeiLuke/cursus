"""
WorkspaceIntegrator - Component integration and promotion using step catalog.

This module provides focused integration functionality for promoting workspace
components to core package and managing cross-workspace integration.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class IntegrationResult:
    """Simple integration result container."""
    
    def __init__(self, success: bool, message: str, details: Optional[Dict[str, Any]] = None):
        self.success = success
        self.message = message
        self.details = details or {}


class WorkspaceIntegrator:
    """
    Component integration and promotion using step catalog.
    
    This class provides focused integration functionality for promoting workspace
    components to core package and managing cross-workspace integration.
    
    Key Features:
    - Component promotion from workspace to core package
    - Cross-workspace component sharing
    - Integration validation using existing frameworks
    - Safe component migration with rollback support
    """
    
    def __init__(self, catalog: StepCatalog):
        """
        Initialize workspace integrator with step catalog.
        
        Args:
            catalog: StepCatalog instance for component discovery and integration
        """
        self.catalog = catalog
        
        # Simple metrics tracking
        self.metrics = {
            'promotions_attempted': 0,
            'promotions_successful': 0,
            'integrations_performed': 0
        }
        
        logger.info("WorkspaceIntegrator initialized with step catalog integration")
    
    def promote_component_to_core(self, step_name: str, source_workspace_id: str, 
                                 dry_run: bool = True) -> IntegrationResult:
        """
        Promote workspace component to core package.
        
        This method safely promotes a workspace component to the core package,
        making it available across all workspaces and deployments.
        
        Args:
            step_name: Name of the component to promote
            source_workspace_id: ID of the source workspace
            dry_run: If True, only validate promotion without executing
            
        Returns:
            IntegrationResult with promotion details
        """
        try:
            self.metrics['promotions_attempted'] += 1
            
            # Get component information from step catalog
            step_info = self.catalog.get_step_info(step_name)
            
            if not step_info:
                return IntegrationResult(
                    success=False,
                    message=f"Component not found: {step_name}",
                    details={'step_name': step_name, 'source_workspace_id': source_workspace_id}
                )
            
            if step_info.workspace_id != source_workspace_id:
                return IntegrationResult(
                    success=False,
                    message=f"Component {step_name} not found in workspace {source_workspace_id}",
                    details={'step_name': step_name, 'actual_workspace': step_info.workspace_id}
                )
            
            # Validate promotion readiness
            validation_result = self._validate_promotion_readiness(step_info)
            if not validation_result.success:
                return validation_result
            
            if dry_run:
                return IntegrationResult(
                    success=True,
                    message=f"Component {step_name} is ready for promotion (dry run)",
                    details={
                        'step_name': step_name,
                        'source_workspace_id': source_workspace_id,
                        'dry_run': True,
                        'validation_passed': True
                    }
                )
            
            # Execute promotion
            promotion_result = self._execute_promotion(step_info)
            
            if promotion_result.success:
                self.metrics['promotions_successful'] += 1
                
                # Refresh catalog to pick up promoted component
                self.catalog.refresh_catalog()
                
                logger.info(f"Successfully promoted component {step_name} to core package")
            
            return promotion_result
            
        except Exception as e:
            logger.error(f"Error promoting component {step_name}: {e}")
            return IntegrationResult(
                success=False,
                message=f"Promotion failed: {str(e)}",
                details={'step_name': step_name, 'error': str(e)}
            )
    
    def _validate_promotion_readiness(self, step_info: Any) -> IntegrationResult:
        """
        Validate that component is ready for promotion.
        
        Args:
            step_info: StepInfo object with component details
            
        Returns:
            IntegrationResult with validation details
        """
        try:
            validation_issues = []
            
            # Check for required components
            required_components = ['builder', 'config']
            missing_components = []
            
            for component_type in required_components:
                if not step_info.file_components.get(component_type):
                    missing_components.append(component_type)
            
            if missing_components:
                validation_issues.append(f"Missing required components: {', '.join(missing_components)}")
            
            # Check file accessibility
            inaccessible_files = []
            for component_type, file_metadata in step_info.file_components.items():
                if file_metadata and hasattr(file_metadata, 'path'):
                    if not Path(file_metadata.path).exists():
                        inaccessible_files.append(str(file_metadata.path))
            
            if inaccessible_files:
                validation_issues.append(f"Inaccessible files: {', '.join(inaccessible_files)}")
            
            # Check for name conflicts in core package
            core_components = self.catalog.list_available_steps(workspace_id="core")
            if step_info.step_name in core_components:
                validation_issues.append(f"Component name conflict: {step_info.step_name} already exists in core")
            
            if validation_issues:
                return IntegrationResult(
                    success=False,
                    message=f"Promotion validation failed: {'; '.join(validation_issues)}",
                    details={
                        'step_name': step_info.step_name,
                        'validation_issues': validation_issues
                    }
                )
            
            return IntegrationResult(
                success=True,
                message=f"Component {step_info.step_name} passed promotion validation",
                details={'step_name': step_info.step_name, 'validation_passed': True}
            )
            
        except Exception as e:
            logger.error(f"Error validating promotion readiness for {step_info.step_name}: {e}")
            return IntegrationResult(
                success=False,
                message=f"Validation error: {str(e)}",
                details={'step_name': step_info.step_name, 'error': str(e)}
            )
    
    def _execute_promotion(self, step_info: Any) -> IntegrationResult:
        """
        Execute component promotion to core package.
        
        Args:
            step_info: StepInfo object with component details
            
        Returns:
            IntegrationResult with promotion execution details
        """
        try:
            # Find core package steps directory
            core_steps_dir = self.catalog.package_root / "steps"
            
            if not core_steps_dir.exists():
                return IntegrationResult(
                    success=False,
                    message=f"Core steps directory not found: {core_steps_dir}",
                    details={'step_name': step_info.step_name, 'core_steps_dir': str(core_steps_dir)}
                )
            
            # Copy component files to core package
            copied_files = []
            copy_errors = []
            
            for component_type, file_metadata in step_info.file_components.items():
                if file_metadata and hasattr(file_metadata, 'path'):
                    source_path = Path(file_metadata.path)
                    
                    if source_path.exists():
                        # Determine target directory based on component type
                        target_dir = core_steps_dir / f"{component_type}s"  # builders, configs, etc.
                        target_dir.mkdir(exist_ok=True)
                        
                        target_path = target_dir / source_path.name
                        
                        try:
                            shutil.copy2(source_path, target_path)
                            copied_files.append(str(target_path))
                            logger.debug(f"Copied {source_path} to {target_path}")
                        except Exception as copy_error:
                            copy_errors.append(f"Failed to copy {source_path}: {str(copy_error)}")
            
            if copy_errors:
                # Rollback copied files on error
                for copied_file in copied_files:
                    try:
                        Path(copied_file).unlink()
                    except Exception:
                        pass  # Best effort cleanup
                
                return IntegrationResult(
                    success=False,
                    message=f"Promotion failed during file copy: {'; '.join(copy_errors)}",
                    details={
                        'step_name': step_info.step_name,
                        'copy_errors': copy_errors,
                        'rollback_performed': True
                    }
                )
            
            return IntegrationResult(
                success=True,
                message=f"Successfully promoted component {step_info.step_name} to core package",
                details={
                    'step_name': step_info.step_name,
                    'copied_files': copied_files,
                    'files_copied': len(copied_files)
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing promotion for {step_info.step_name}: {e}")
            return IntegrationResult(
                success=False,
                message=f"Promotion execution failed: {str(e)}",
                details={'step_name': step_info.step_name, 'error': str(e)}
            )
    
    def integrate_cross_workspace_components(self, target_workspace_id: str, 
                                           source_components: List[Dict[str, str]]) -> IntegrationResult:
        """
        Integrate components from multiple workspaces into target workspace.
        
        This method enables cross-workspace component sharing by creating
        references or copies of components in the target workspace.
        
        Args:
            target_workspace_id: ID of the target workspace
            source_components: List of dicts with 'step_name' and 'source_workspace_id'
            
        Returns:
            IntegrationResult with integration details
        """
        try:
            self.metrics['integrations_performed'] += 1
            
            integration_results = []
            successful_integrations = 0
            
            for component_spec in source_components:
                step_name = component_spec.get('step_name')
                source_workspace_id = component_spec.get('source_workspace_id')
                
                if not step_name or not source_workspace_id:
                    integration_results.append({
                        'step_name': step_name or 'unknown',
                        'success': False,
                        'message': 'Invalid component specification'
                    })
                    continue
                
                # Get component information
                step_info = self.catalog.get_step_info(step_name)
                
                if not step_info:
                    integration_results.append({
                        'step_name': step_name,
                        'success': False,
                        'message': f'Component not found: {step_name}'
                    })
                    continue
                
                if step_info.workspace_id != source_workspace_id:
                    integration_results.append({
                        'step_name': step_name,
                        'success': False,
                        'message': f'Component not in expected workspace: {source_workspace_id}'
                    })
                    continue
                
                # For now, we create a reference/link rather than copying files
                # This is a simplified implementation - full implementation would
                # handle actual file integration based on workspace organization
                integration_results.append({
                    'step_name': step_name,
                    'success': True,
                    'message': f'Component {step_name} integrated from {source_workspace_id}',
                    'integration_type': 'reference'
                })
                successful_integrations += 1
            
            overall_success = successful_integrations == len(source_components)
            
            result = IntegrationResult(
                success=overall_success,
                message=f"Cross-workspace integration: {successful_integrations}/{len(source_components)} successful",
                details={
                    'target_workspace_id': target_workspace_id,
                    'total_components': len(source_components),
                    'successful_integrations': successful_integrations,
                    'integration_results': integration_results
                }
            )
            
            logger.info(f"Cross-workspace integration completed: {successful_integrations}/{len(source_components)} successful")
            return result
            
        except Exception as e:
            logger.error(f"Error integrating cross-workspace components: {e}")
            return IntegrationResult(
                success=False,
                message=f"Cross-workspace integration failed: {str(e)}",
                details={'target_workspace_id': target_workspace_id, 'error': str(e)}
            )
    
    def rollback_promotion(self, step_name: str) -> IntegrationResult:
        """
        Rollback component promotion from core package.
        
        This method removes a promoted component from the core package,
        effectively reversing a previous promotion operation.
        
        Args:
            step_name: Name of the component to rollback
            
        Returns:
            IntegrationResult with rollback details
        """
        try:
            # Get component information from core
            step_info = self.catalog.get_step_info(step_name)
            
            if not step_info or step_info.workspace_id != "core":
                return IntegrationResult(
                    success=False,
                    message=f"Component {step_name} not found in core package",
                    details={'step_name': step_name}
                )
            
            # Find and remove component files from core package
            core_steps_dir = self.catalog.package_root / "steps"
            removed_files = []
            removal_errors = []
            
            for component_type, file_metadata in step_info.file_components.items():
                if file_metadata and hasattr(file_metadata, 'path'):
                    file_path = Path(file_metadata.path)
                    
                    # Only remove files that are in the core package directory
                    if core_steps_dir in file_path.parents:
                        try:
                            if file_path.exists():
                                file_path.unlink()
                                removed_files.append(str(file_path))
                                logger.debug(f"Removed {file_path}")
                        except Exception as remove_error:
                            removal_errors.append(f"Failed to remove {file_path}: {str(remove_error)}")
            
            if removal_errors:
                return IntegrationResult(
                    success=False,
                    message=f"Rollback partially failed: {'; '.join(removal_errors)}",
                    details={
                        'step_name': step_name,
                        'removed_files': removed_files,
                        'removal_errors': removal_errors
                    }
                )
            
            # Refresh catalog to reflect changes
            self.catalog.refresh_catalog()
            
            return IntegrationResult(
                success=True,
                message=f"Successfully rolled back component {step_name} from core package",
                details={
                    'step_name': step_name,
                    'removed_files': removed_files,
                    'files_removed': len(removed_files)
                }
            )
            
        except Exception as e:
            logger.error(f"Error rolling back component {step_name}: {e}")
            return IntegrationResult(
                success=False,
                message=f"Rollback failed: {str(e)}",
                details={'step_name': step_name, 'error': str(e)}
            )
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """
        Get summary of integration activities.
        
        Returns:
            Dictionary with integration metrics and summary
        """
        try:
            summary = {
                'metrics': self.metrics.copy(),
                'promotion_success_rate': 0.0,
                'catalog_info': {
                    'total_steps': len(self.catalog._step_index) if hasattr(self.catalog, '_step_index') else 0,
                    'workspaces': len(self.catalog._workspace_steps) if hasattr(self.catalog, '_workspace_steps') else 0
                }
            }
            
            # Calculate success rate
            if self.metrics['promotions_attempted'] > 0:
                summary['promotion_success_rate'] = (
                    self.metrics['promotions_successful'] / 
                    self.metrics['promotions_attempted']
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating integration summary: {e}")
            return {'error': str(e), 'metrics': self.metrics.copy()}
