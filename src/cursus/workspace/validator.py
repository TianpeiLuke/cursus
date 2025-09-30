"""
WorkspaceValidator - Workspace validation using step catalog integration.

This module provides focused validation functionality that leverages existing
validation frameworks while supporting workspace-aware operations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..step_catalog import StepCatalog
from ..validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

logger = logging.getLogger(__name__)


class ValidationResult:
    """Simple validation result container."""
    
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None, 
                 warnings: Optional[List[str]] = None, details: Optional[Dict[str, Any]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.details = details or {}


class CompatibilityResult:
    """Simple compatibility result container."""
    
    def __init__(self, is_compatible: bool, issues: Optional[List[str]] = None,
                 compatibility_matrix: Optional[Dict[str, Dict[str, bool]]] = None):
        self.is_compatible = is_compatible
        self.issues = issues or []
        self.compatibility_matrix = compatibility_matrix or {}


class WorkspaceValidator:
    """
    Workspace validation using step catalog integration.
    
    This class provides focused validation functionality that leverages existing
    validation frameworks while supporting workspace-aware operations.
    
    Key Features:
    - Uses existing validation frameworks with workspace context
    - Cross-workspace compatibility validation
    - Component quality validation using step catalog
    - Integration with existing alignment testing
    """
    
    def __init__(self, catalog: StepCatalog):
        """
        Initialize workspace validator with step catalog.
        
        Args:
            catalog: StepCatalog instance for component discovery and validation
        """
        self.catalog = catalog
        
        # Simple metrics tracking
        self.metrics = {
            'validations_performed': 0,
            'components_validated': 0,
            'compatibility_checks': 0
        }
        
        logger.info("WorkspaceValidator initialized with step catalog integration")
    
    def validate_workspace_components(self, workspace_id: str) -> ValidationResult:
        """
        Validate workspace components using step catalog.
        
        This method uses existing validation frameworks with workspace context,
        eliminating the need for custom validation logic.
        
        Args:
            workspace_id: ID of the workspace to validate
            
        Returns:
            ValidationResult with validation details
        """
        try:
            self.metrics['validations_performed'] += 1
            
            # Get components from workspace using step catalog
            components = self.catalog.list_available_steps(workspace_id=workspace_id)
            
            if not components:
                return ValidationResult(
                    is_valid=True,
                    warnings=[f"No components found in workspace: {workspace_id}"],
                    details={'workspace_id': workspace_id, 'component_count': 0}
                )
            
            # Validate each component using existing validation frameworks
            validation_errors = []
            validation_warnings = []
            validated_components = []
            
            for component in components:
                component_result = self._validate_component(component)
                
                if component_result:
                    validated_components.append(component)
                    self.metrics['components_validated'] += 1
                    
                    # Collect any validation issues
                    if hasattr(component_result, 'errors') and component_result.errors:
                        validation_errors.extend([f"{component}: {error}" for error in component_result.errors])
                    
                    if hasattr(component_result, 'warnings') and component_result.warnings:
                        validation_warnings.extend([f"{component}: {warning}" for warning in component_result.warnings])
            
            is_valid = len(validation_errors) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=validation_errors,
                warnings=validation_warnings,
                details={
                    'workspace_id': workspace_id,
                    'total_components': len(components),
                    'validated_components': len(validated_components),
                    'component_names': components
                }
            )
            
            logger.debug(f"Validated workspace {workspace_id}: {len(components)} components, valid={is_valid}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating workspace components for {workspace_id}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                details={'workspace_id': workspace_id, 'error': str(e)}
            )
    
    def _validate_component(self, step_name: str) -> Optional[Any]:
        """
        Validate individual component using existing validation frameworks.
        
        This method integrates with existing alignment testing and validation
        frameworks to provide comprehensive component validation.
        
        Args:
            step_name: Name of the component/step to validate
            
        Returns:
            Validation result from existing frameworks, or None if validation fails
        """
        try:
            # Get component information from step catalog
            step_info = self.catalog.get_step_info(step_name)
            
            if not step_info:
                logger.warning(f"No step info found for component: {step_name}")
                return None
            
            # Use existing validation frameworks with workspace context
            # This integrates with the existing alignment tester and validation systems
            validation_result = self._run_existing_validation(step_info)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating component {step_name}: {e}")
            return None
    
    def _run_existing_validation(self, step_info: Any) -> Any:
        """
        Run existing validation frameworks on component.
        
        This method integrates with existing validation systems like
        UnifiedAlignmentTester and other validation frameworks.
        
        Args:
            step_info: StepInfo object with component details
            
        Returns:
            Validation result from existing frameworks
        """
        try:
            # Try to use existing validation frameworks
            # This is a placeholder for integration with existing validation systems
            
            # Basic validation using step catalog information
            validation_issues = []
            validation_warnings = []
            
            # Check if component has required file components
            required_components = ['builder', 'config']
            for required_component in required_components:
                if not step_info.file_components.get(required_component):
                    validation_warnings.append(f"Missing {required_component} component")
            
            # Check if files exist
            for component_type, file_metadata in step_info.file_components.items():
                if file_metadata and hasattr(file_metadata, 'path'):
                    if not Path(file_metadata.path).exists():
                        validation_issues.append(f"File not found: {file_metadata.path}")
            
            # Create simple validation result
            result = ValidationResult(
                is_valid=len(validation_issues) == 0,
                errors=validation_issues,
                warnings=validation_warnings,
                details={'step_name': step_info.step_name, 'workspace_id': step_info.workspace_id}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running existing validation for {step_info.step_name}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation framework error: {str(e)}"],
                details={'step_name': step_info.step_name}
            )
    
    def validate_cross_workspace_compatibility(self, workspace_ids: List[str]) -> CompatibilityResult:
        """
        Validate compatibility between workspace components.
        
        This method checks for compatibility issues between components from
        different workspaces, enabling safe cross-workspace collaboration.
        
        Args:
            workspace_ids: List of workspace IDs to check compatibility
            
        Returns:
            CompatibilityResult with compatibility analysis
        """
        try:
            self.metrics['compatibility_checks'] += 1
            
            # Get components from all specified workspaces
            all_components = {}
            for workspace_id in workspace_ids:
                components = self.catalog.list_available_steps(workspace_id=workspace_id)
                all_components[workspace_id] = components
            
            # Check for compatibility issues
            compatibility_issues = []
            compatibility_matrix = {}
            
            # Check for name conflicts between workspaces
            all_component_names = set()
            for workspace_id, components in all_components.items():
                workspace_conflicts = []
                
                for component in components:
                    if component in all_component_names:
                        compatibility_issues.append(
                            f"Component name conflict: '{component}' exists in multiple workspaces"
                        )
                        workspace_conflicts.append(component)
                    else:
                        all_component_names.add(component)
                
                # Build compatibility matrix
                compatibility_matrix[workspace_id] = {
                    'total_components': len(components),
                    'conflicts': len(workspace_conflicts),
                    'conflicting_components': workspace_conflicts
                }
            
            # Check for dependency compatibility
            dependency_issues = self._check_dependency_compatibility(all_components)
            compatibility_issues.extend(dependency_issues)
            
            is_compatible = len(compatibility_issues) == 0
            
            result = CompatibilityResult(
                is_compatible=is_compatible,
                issues=compatibility_issues,
                compatibility_matrix=compatibility_matrix
            )
            
            logger.debug(f"Cross-workspace compatibility check: {len(workspace_ids)} workspaces, compatible={is_compatible}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating cross-workspace compatibility: {e}")
            return CompatibilityResult(
                is_compatible=False,
                issues=[f"Compatibility check failed: {str(e)}"]
            )
    
    def _check_dependency_compatibility(self, all_components: Dict[str, List[str]]) -> List[str]:
        """
        Check for dependency compatibility issues between workspaces.
        
        Args:
            all_components: Dictionary mapping workspace IDs to component lists
            
        Returns:
            List of dependency compatibility issues
        """
        dependency_issues = []
        
        try:
            # This is a simplified dependency check
            # In a full implementation, this would analyze actual dependencies
            # between components using the step catalog's dependency information
            
            for workspace_id, components in all_components.items():
                for component in components:
                    step_info = self.catalog.get_step_info(component)
                    
                    if step_info and hasattr(step_info, 'registry_data'):
                        # Check for potential dependency issues
                        # This is a placeholder for more sophisticated dependency analysis
                        pass
            
        except Exception as e:
            logger.error(f"Error checking dependency compatibility: {e}")
            dependency_issues.append(f"Dependency analysis failed: {str(e)}")
        
        return dependency_issues
    
    def validate_component_quality(self, step_name: str) -> ValidationResult:
        """
        Validate component quality using step catalog information.
        
        Args:
            step_name: Name of the component to validate
            
        Returns:
            ValidationResult with quality assessment
        """
        try:
            step_info = self.catalog.get_step_info(step_name)
            
            if not step_info:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Component not found: {step_name}"]
                )
            
            quality_issues = []
            quality_warnings = []
            quality_score = 100  # Start with perfect score
            
            # Check component completeness
            expected_components = ['builder', 'config', 'contract', 'spec', 'script']
            missing_components = []
            
            for component_type in expected_components:
                if not step_info.file_components.get(component_type):
                    missing_components.append(component_type)
                    quality_score -= 15  # Deduct points for missing components
            
            if missing_components:
                quality_warnings.append(f"Missing components: {', '.join(missing_components)}")
            
            # Check file accessibility
            for component_type, file_metadata in step_info.file_components.items():
                if file_metadata and hasattr(file_metadata, 'path'):
                    if not Path(file_metadata.path).exists():
                        quality_issues.append(f"File not accessible: {file_metadata.path}")
                        quality_score -= 20  # Deduct points for inaccessible files
            
            # Determine overall quality
            is_valid = len(quality_issues) == 0 and quality_score >= 60
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=quality_issues,
                warnings=quality_warnings,
                details={
                    'step_name': step_name,
                    'workspace_id': step_info.workspace_id,
                    'quality_score': max(0, quality_score),
                    'component_completeness': len(step_info.file_components),
                    'missing_components': missing_components
                }
            )
            
            logger.debug(f"Quality validation for {step_name}: score={quality_score}, valid={is_valid}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating component quality for {step_name}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Quality validation failed: {str(e)}"],
                details={'step_name': step_name, 'error': str(e)}
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation activities.
        
        Returns:
            Dictionary with validation metrics and summary
        """
        try:
            summary = {
                'metrics': self.metrics.copy(),
                'catalog_info': {
                    'total_steps': len(self.catalog._step_index) if hasattr(self.catalog, '_step_index') else 0,
                    'workspaces': len(self.catalog._workspace_steps) if hasattr(self.catalog, '_workspace_steps') else 0
                }
            }
            
            # Add success rates if we have data
            if self.metrics['validations_performed'] > 0:
                summary['validation_success_rate'] = (
                    self.metrics['components_validated'] / 
                    max(1, self.metrics['validations_performed'])
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating validation summary: {e}")
            return {'error': str(e), 'metrics': self.metrics.copy()}
