"""
Legacy Adapters

Backward compatibility helpers for existing workspace validation APIs.
This component ensures that existing code continues to work unchanged
while providing access to the new unified validation system.

PHASE 4 DEPRECATION NOTICE:
These legacy adapters are planned for deprecation in a future release.
Please migrate to the unified validation system for new code.

Architecture:
- Adapter pattern for existing API compatibility
- Transparent integration with unified validation system
- Gradual migration support for legacy code
- Consistent behavior preservation

Features:
- Full backward compatibility with existing APIs
- Transparent access to unified validation features
- Gradual migration path for legacy code
- Consistent result format preservation
"""

from typing import Dict, List, Any, Optional, Union
import logging
import warnings
from pathlib import Path

from .unified_validation_core import UnifiedValidationCore, ValidationConfig
from .unified_result_structures import UnifiedValidationResult
from .unified_report_generator import UnifiedReportGenerator, ReportConfig
# PHASE 1 CONSOLIDATION: WorkspaceValidationOrchestrator removed, using WorkspaceTestManager
from .workspace_test_manager import WorkspaceTestManager

logger = logging.getLogger(__name__)


class LegacyWorkspaceValidationAdapter:
    """
    Adapter for legacy workspace validation APIs.
    
    This adapter provides backward compatibility for existing workspace
    validation code while transparently using the new unified validation
    system underneath.
    
    Features:
    - Maintains existing API signatures
    - Converts unified results to legacy format
    - Preserves existing behavior patterns
    - Enables gradual migration to unified system
    """
    
    def __init__(
        self,
        workspace_root: Union[str, Path],
        enable_unified_features: bool = False
    ):
        """
        Initialize legacy workspace validation adapter.
        
        PHASE 4 DEPRECATION WARNING: This adapter is planned for deprecation.
        Please migrate to the unified validation system for new code.
        
        Args:
            workspace_root: Root directory containing workspace(s)
            enable_unified_features: Whether to enable new unified features
        """
        # PHASE 4: Issue deprecation warning
        warnings.warn(
            "LegacyWorkspaceValidationAdapter is deprecated and will be removed in a future release. "
            "Please migrate to the unified validation system (UnifiedValidationCore).",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.workspace_root = Path(workspace_root)
        self.enable_unified_features = enable_unified_features
        
        # Initialize unified validation core
        self.unified_core = UnifiedValidationCore(workspace_root)
        
        # PHASE 1 CONSOLIDATION: Initialize WorkspaceTestManager for compatibility
        self.legacy_orchestrator = None
        try:
            self.legacy_orchestrator = WorkspaceTestManager(
                workspace_manager=None,  # Will create default
                enable_parallel_validation=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize legacy test manager: {e}")
    
    def validate_workspace(
        self,
        developer_id: str,
        validation_levels: Optional[List[str]] = None,
        target_scripts: Optional[List[str]] = None,
        target_builders: Optional[List[str]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Legacy API: Validate a single workspace.
        
        This method maintains the existing API signature while using
        the unified validation system underneath.
        
        Args:
            developer_id: Developer workspace to validate
            validation_levels: Validation types to run
            target_scripts: Specific scripts to validate
            target_builders: Specific builders to validate
            validation_config: Additional validation configuration
            
        Returns:
            Legacy-format validation results
        """
        try:
            # Convert legacy parameters to unified config
            unified_config = self._convert_legacy_config(
                validation_levels=validation_levels,
                target_scripts=target_scripts,
                target_builders=target_builders,
                validation_config=validation_config
            )
            
            # Run unified validation
            unified_result = self.unified_core.validate_workspaces(
                validation_config=unified_config
            )
            
            # Convert unified result to legacy format
            legacy_result = self._convert_to_legacy_single_workspace_result(
                unified_result, developer_id
            )
            
            return legacy_result
            
        except Exception as e:
            logger.error(f"Legacy workspace validation failed: {e}")
            
            # Fallback to legacy orchestrator if available
            if self.legacy_orchestrator:
                try:
                    return self.legacy_orchestrator.validate_workspace(
                        developer_id=developer_id,
                        validation_levels=validation_levels,
                        target_scripts=target_scripts,
                        target_builders=target_builders,
                        validation_config=validation_config
                    )
                except Exception as fallback_error:
                    logger.error(f"Legacy orchestrator fallback failed: {fallback_error}")
            
            # Return error result in legacy format
            return {
                'developer_id': developer_id,
                'success': False,
                'error': str(e),
                'results': {},
                'summary': {'error': 'Validation failed to complete'},
                'recommendations': ['Fix validation setup issues before retrying']
            }
    
    def validate_all_workspaces(
        self,
        validation_levels: Optional[List[str]] = None,
        parallel: Optional[bool] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Legacy API: Validate all workspaces.
        
        This method maintains the existing API signature while using
        the unified validation system underneath.
        
        Args:
            validation_levels: Validation types to run
            parallel: Whether to run validations in parallel
            validation_config: Additional validation configuration
            
        Returns:
            Legacy-format multi-workspace validation results
        """
        try:
            # Convert legacy parameters to unified config
            unified_config = self._convert_legacy_config(
                validation_levels=validation_levels,
                validation_config=validation_config
            )
            
            # Set parallel validation if specified
            if parallel is not None:
                unified_config.parallel_validation = parallel
            
            # Run unified validation
            unified_result = self.unified_core.validate_workspaces(
                validation_config=unified_config
            )
            
            # Convert unified result to legacy format
            legacy_result = self._convert_to_legacy_multi_workspace_result(
                unified_result
            )
            
            return legacy_result
            
        except Exception as e:
            logger.error(f"Legacy multi-workspace validation failed: {e}")
            
            # Fallback to legacy test manager if available
            if self.legacy_orchestrator:
                try:
                    # WorkspaceTestManager doesn't have validate_all_workspaces, 
                    # so we'll need to implement multi-workspace logic here
                    logger.warning("Multi-workspace validation fallback not fully implemented for WorkspaceTestManager")
                    return {
                        'workspace_root': str(self.workspace_root),
                        'total_workspaces': 0,
                        'successful_validations': 0,
                        'failed_validations': 0,
                        'success': False,
                        'error': 'Multi-workspace validation not supported in fallback mode',
                        'results': {},
                        'summary': {'error': 'Multi-workspace validation fallback not implemented'},
                        'recommendations': ['Use unified validation system for multi-workspace scenarios']
                    }
                except Exception as fallback_error:
                    logger.error(f"Legacy test manager fallback failed: {fallback_error}")
            
            # Return error result in legacy format
            return {
                'workspace_root': str(self.workspace_root),
                'total_workspaces': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'success': False,
                'error': str(e),
                'results': {},
                'summary': {'error': 'Multi-workspace validation failed to complete'},
                'recommendations': ['Fix validation setup issues before retrying']
            }
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy API: Generate validation report.
        
        This method maintains compatibility with existing report generation
        while optionally using the new unified report generator.
        
        Args:
            validation_results: Validation results to generate report for
            
        Returns:
            Legacy-format validation report
        """
        try:
            if self.enable_unified_features:
                # Try to convert legacy results to unified format and use new generator
                unified_result = self._convert_legacy_to_unified_result(validation_results)
                if unified_result:
                    report_generator = UnifiedReportGenerator()
                    unified_report = report_generator.generate_report(unified_result)
                    return self._convert_unified_to_legacy_report(unified_report)
            
            # Fallback to legacy orchestrator report generation
            if self.legacy_orchestrator:
                return self.legacy_orchestrator.generate_validation_report(validation_results)
            
            # Basic legacy report format
            return {
                'summary': self._extract_legacy_summary(validation_results),
                'details': validation_results,
                'recommendations': validation_results.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Legacy report generation failed: {e}")
            return {
                'summary': {'error': 'Report generation failed'},
                'details': validation_results,
                'recommendations': ['Unable to generate detailed report']
            }
    
    def _convert_legacy_config(
        self,
        validation_levels: Optional[List[str]] = None,
        target_scripts: Optional[List[str]] = None,
        target_builders: Optional[List[str]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationConfig:
        """Convert legacy configuration parameters to unified config."""
        
        # Default validation levels
        if validation_levels is None:
            validation_levels = ['alignment', 'builders']
        
        # Extract additional config parameters
        config_dict = validation_config or {}
        
        return ValidationConfig(
            validation_types=validation_levels,
            target_scripts=target_scripts,
            target_builders=target_builders,
            skip_levels=config_dict.get('skip_levels'),
            strict_validation=config_dict.get('strict_validation', False),
            parallel_validation=config_dict.get('parallel_validation', True),
            max_workers=config_dict.get('max_workers'),
            timeout_seconds=config_dict.get('timeout_seconds'),
            workspace_context=config_dict.get('workspace_context', {})
        )
    
    def _convert_to_legacy_single_workspace_result(
        self,
        unified_result: UnifiedValidationResult,
        developer_id: str
    ) -> Dict[str, Any]:
        """Convert unified result to legacy single workspace format."""
        
        # Find the workspace result (could be "default" for single workspace)
        workspace_result = None
        if developer_id in unified_result.workspaces:
            workspace_result = unified_result.workspaces[developer_id]
        elif "default" in unified_result.workspaces:
            workspace_result = unified_result.workspaces["default"]
        elif len(unified_result.workspaces) == 1:
            workspace_result = list(unified_result.workspaces.values())[0]
        
        if not workspace_result:
            return {
                'developer_id': developer_id,
                'success': False,
                'error': 'Workspace not found in validation results',
                'results': {},
                'summary': {'error': 'Workspace validation failed'},
                'recommendations': ['Check workspace configuration']
            }
        
        # Convert to legacy format
        legacy_result = {
            'developer_id': developer_id,
            'workspace_root': unified_result.workspace_root,
            'success': workspace_result.success,
            'results': {},
            'summary': {},
            'recommendations': unified_result.recommendations
        }
        
        # Add timing information
        if workspace_result.validation_start_time:
            legacy_result['validation_start_time'] = workspace_result.validation_start_time
        if workspace_result.validation_end_time:
            legacy_result['validation_end_time'] = workspace_result.validation_end_time
        if workspace_result.validation_duration_seconds:
            legacy_result['validation_duration_seconds'] = workspace_result.validation_duration_seconds
        
        # Add validation results
        if workspace_result.alignment_results:
            legacy_result['results']['alignment'] = workspace_result.alignment_results
        
        if workspace_result.builder_results:
            legacy_result['results']['builders'] = workspace_result.builder_results
        
        # Add error information
        if workspace_result.error:
            legacy_result['error'] = workspace_result.error
        
        if workspace_result.warnings:
            legacy_result['warnings'] = workspace_result.warnings
        
        # Generate summary
        legacy_result['summary'] = {
            'validation_types_run': workspace_result.validation_types_run,
            'overall_success': workspace_result.success,
            'has_alignment_results': workspace_result.has_alignment_results,
            'has_builder_results': workspace_result.has_builder_results
        }
        
        return legacy_result
    
    def _convert_to_legacy_multi_workspace_result(
        self,
        unified_result: UnifiedValidationResult
    ) -> Dict[str, Any]:
        """Convert unified result to legacy multi-workspace format."""
        
        legacy_result = {
            'workspace_root': unified_result.workspace_root,
            'total_workspaces': unified_result.summary.total_workspaces,
            'validated_workspaces': unified_result.summary.total_workspaces,
            'successful_validations': unified_result.summary.successful_workspaces,
            'failed_validations': unified_result.summary.failed_workspaces,
            'success_rate': unified_result.summary.success_rate,
            'success': unified_result.overall_success,
            'results': {},
            'summary': {},
            'recommendations': unified_result.recommendations
        }
        
        # Add timing information
        if unified_result.validation_start_time:
            legacy_result['validation_start_time'] = unified_result.validation_start_time
        if unified_result.validation_end_time:
            legacy_result['validation_end_time'] = unified_result.validation_end_time
        if unified_result.validation_duration_seconds:
            legacy_result['validation_duration_seconds'] = unified_result.validation_duration_seconds
        
        # Add global error if present
        if unified_result.global_error:
            legacy_result['error'] = unified_result.global_error
        
        # Convert workspace results
        for workspace_id, workspace_result in unified_result.workspaces.items():
            legacy_workspace_result = {
                'success': workspace_result.success,
                'workspace_type': workspace_result.workspace_type,
                'workspace_path': workspace_result.workspace_path,
                'results': {}
            }
            
            # Add validation results
            if workspace_result.alignment_results:
                legacy_workspace_result['results']['alignment'] = workspace_result.alignment_results
            
            if workspace_result.builder_results:
                legacy_workspace_result['results']['builders'] = workspace_result.builder_results
            
            # Add error information
            if workspace_result.error:
                legacy_workspace_result['error'] = workspace_result.error
            
            if workspace_result.warnings:
                legacy_workspace_result['warnings'] = workspace_result.warnings
            
            legacy_result['results'][workspace_id] = legacy_workspace_result
        
        # Generate summary compatible with legacy expectations
        legacy_result['summary'] = {
            'overall_statistics': {
                'total_workspaces': unified_result.summary.total_workspaces,
                'successful_workspaces': unified_result.summary.successful_workspaces,
                'failed_workspaces': unified_result.summary.failed_workspaces,
                'success_rate': unified_result.summary.success_rate
            },
            'workspace_details': {
                workspace_id: {
                    'success': workspace_result.success,
                    'validation_types': workspace_result.validation_types_run,
                    'has_error': workspace_result.error is not None
                }
                for workspace_id, workspace_result in unified_result.workspaces.items()
            }
        }
        
        return legacy_result
    
    def _convert_legacy_to_unified_result(
        self,
        legacy_results: Dict[str, Any]
    ) -> Optional[UnifiedValidationResult]:
        """Convert legacy results to unified format (best effort)."""
        try:
            # This is a complex conversion that may not be perfect
            # It's mainly for enabling unified features on legacy results
            
            # Determine if this is single or multi-workspace legacy result
            if 'developer_id' in legacy_results:
                # Single workspace legacy result
                # This conversion is complex and may not be fully accurate
                # For now, return None to use legacy processing
                return None
            else:
                # Multi-workspace legacy result
                # This conversion is complex and may not be fully accurate
                # For now, return None to use legacy processing
                return None
                
        except Exception as e:
            logger.warning(f"Failed to convert legacy results to unified format: {e}")
            return None
    
    def _convert_unified_to_legacy_report(
        self,
        unified_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert unified report to legacy format."""
        
        # Extract key information from unified report
        legacy_report = {
            'summary': {},
            'details': unified_report,
            'recommendations': unified_report.get('recommendations', [])
        }
        
        # Convert summary based on report type
        if unified_report.get('report_type') == 'single_workspace_validation':
            validation_summary = unified_report.get('validation_summary', {})
            legacy_report['summary'] = {
                'overall_success': validation_summary.get('overall_success', False),
                'validation_types_run': validation_summary.get('validation_types_run', []),
                'has_errors': validation_summary.get('has_errors', False),
                'warning_count': validation_summary.get('warning_count', 0)
            }
        else:
            validation_summary = unified_report.get('validation_summary', {})
            legacy_report['summary'] = {
                'total_workspaces': validation_summary.get('total_workspaces', 0),
                'successful_workspaces': validation_summary.get('successful_workspaces', 0),
                'failed_workspaces': validation_summary.get('failed_workspaces', 0),
                'success_rate': validation_summary.get('success_rate', 0.0),
                'overall_success': validation_summary.get('overall_success', False)
            }
        
        return legacy_report
    
    def _extract_legacy_summary(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract summary from legacy validation results."""
        
        # Try to extract summary from various legacy formats
        if 'summary' in validation_results:
            return validation_results['summary']
        
        # Generate basic summary from available information
        summary = {}
        
        if 'success' in validation_results:
            summary['overall_success'] = validation_results['success']
        
        if 'total_workspaces' in validation_results:
            summary['total_workspaces'] = validation_results['total_workspaces']
            summary['successful_workspaces'] = validation_results.get('successful_validations', 0)
            summary['failed_workspaces'] = validation_results.get('failed_validations', 0)
            summary['success_rate'] = validation_results.get('success_rate', 0.0)
        
        return summary


def create_legacy_adapter(
    workspace_root: Union[str, Path],
    enable_unified_features: bool = False
) -> LegacyWorkspaceValidationAdapter:
    """
    Convenience function to create a legacy workspace validation adapter.
    
    Args:
        workspace_root: Root directory containing workspace(s)
        enable_unified_features: Whether to enable new unified features
        
    Returns:
        Configured LegacyWorkspaceValidationAdapter
    """
    return LegacyWorkspaceValidationAdapter(
        workspace_root=workspace_root,
        enable_unified_features=enable_unified_features
    )


def validate_workspace_legacy(
    workspace_root: Union[str, Path],
    developer_id: str,
    validation_levels: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Legacy function for single workspace validation.
    
    PHASE 4 DEPRECATION WARNING: This function is planned for deprecation.
    Please migrate to the unified validation system for new code.
    
    This function provides backward compatibility for existing code
    that uses the legacy workspace validation API.
    
    Args:
        workspace_root: Root directory containing workspace(s)
        developer_id: Developer workspace to validate
        validation_levels: Validation types to run
        **kwargs: Additional validation parameters
        
    Returns:
        Legacy-format validation results
    """
    # PHASE 4: Issue deprecation warning
    warnings.warn(
        "validate_workspace_legacy() is deprecated and will be removed in a future release. "
        "Please use UnifiedValidationCore.validate_workspaces() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    adapter = create_legacy_adapter(workspace_root)
    return adapter.validate_workspace(
        developer_id=developer_id,
        validation_levels=validation_levels,
        **kwargs
    )


def validate_all_workspaces_legacy(
    workspace_root: Union[str, Path],
    validation_levels: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Legacy function for multi-workspace validation.
    
    PHASE 4 DEPRECATION WARNING: This function is planned for deprecation.
    Please migrate to the unified validation system for new code.
    
    This function provides backward compatibility for existing code
    that uses the legacy multi-workspace validation API.
    
    Args:
        workspace_root: Root directory containing workspace(s)
        validation_levels: Validation types to run
        **kwargs: Additional validation parameters
        
    Returns:
        Legacy-format multi-workspace validation results
    """
    # PHASE 4: Issue deprecation warning
    warnings.warn(
        "validate_all_workspaces_legacy() is deprecated and will be removed in a future release. "
        "Please use UnifiedValidationCore.validate_workspaces() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    adapter = create_legacy_adapter(workspace_root)
    return adapter.validate_all_workspaces(
        validation_levels=validation_levels,
        **kwargs
    )
