"""
Unified Validation Core

Core validation logic that works identically for single and multi-workspace scenarios.
This component eliminates dual-path complexity by providing a single validation
method that handles both count=1 and count=N scenarios uniformly.

Architecture:
- Single validation method for all scenarios
- Unified workspace entry validation logic
- Consistent error handling and reporting
- Extensible validation framework

Features:
- Single validation method regardless of workspace count
- Consistent validation logic for all workspace types
- Unified error handling and diagnostics
- Extensible validation pipeline architecture
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
import logging

from .workspace_type_detector import WorkspaceTypeDetector
from .unified_result_structures import (
    UnifiedValidationResult,
    WorkspaceValidationResult,
    ValidationResultBuilder,
    create_empty_result
)
from .workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
from .workspace_builder_test import WorkspaceUniversalStepBuilderTest

logger = logging.getLogger(__name__)


class ValidationConfig:
    """Configuration for unified validation operations."""
    
    def __init__(
        self,
        validation_types: Optional[List[str]] = None,
        target_scripts: Optional[List[str]] = None,
        target_builders: Optional[List[str]] = None,
        skip_levels: Optional[List[str]] = None,
        strict_validation: bool = False,
        parallel_validation: bool = True,
        max_workers: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        workspace_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation configuration.
        
        Args:
            validation_types: Types of validation to run ('alignment', 'builders', 'all')
            target_scripts: Specific scripts to validate (None for all)
            target_builders: Specific builders to validate (None for all)
            skip_levels: Validation levels to skip
            strict_validation: Whether to apply strict validation rules
            parallel_validation: Whether to enable parallel validation
            max_workers: Maximum number of parallel workers
            timeout_seconds: Timeout for individual validations
            workspace_context: Additional context for workspace validation
        """
        self.validation_types = validation_types or ['alignment', 'builders']
        self.target_scripts = target_scripts
        self.target_builders = target_builders
        self.skip_levels = skip_levels
        self.strict_validation = strict_validation
        self.parallel_validation = parallel_validation
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.workspace_context = workspace_context or {}


class UnifiedValidationCore:
    """
    Core validation logic that works identically for single and multi-workspace.
    
    This class provides a single validation method that handles both single
    workspace (count=1) and multi-workspace (count=N) scenarios uniformly,
    eliminating the need for separate validation paths.
    
    Features:
    - Single validation method for all scenarios
    - Consistent workspace entry validation
    - Unified error handling and reporting
    - Extensible validation pipeline
    """
    
    def __init__(
        self,
        workspace_root: Union[str, Path],
        validation_config: Optional[ValidationConfig] = None
    ):
        """
        Initialize unified validation core.
        
        Args:
            workspace_root: Root directory containing workspace(s)
            validation_config: Configuration for validation operations
        """
        self.workspace_root = Path(workspace_root)
        self.validation_config = validation_config or ValidationConfig()
        
        # Initialize workspace type detector
        self.workspace_detector = WorkspaceTypeDetector(workspace_root)
        
        # Validation state
        self._detected_workspaces: Optional[Dict[str, Any]] = None
        self._workspace_type: Optional[str] = None
    
    def validate_workspaces(
        self,
        validation_config: Optional[ValidationConfig] = None,
        **kwargs
    ) -> UnifiedValidationResult:
        """
        Single validation method for all scenarios.
        
        This method works identically whether validating a single workspace
        (count=1) or multiple workspaces (count=N), providing a unified
        validation interface.
        
        Args:
            validation_config: Override validation configuration
            **kwargs: Additional validation parameters
            
        Returns:
            UnifiedValidationResult with consistent structure regardless of count
        """
        # Use provided config or instance default
        config = validation_config or self.validation_config
        
        # Override config with kwargs
        if 'validation_types' in kwargs:
            config.validation_types = kwargs['validation_types']
        if 'target_scripts' in kwargs:
            config.target_scripts = kwargs['target_scripts']
        if 'target_builders' in kwargs:
            config.target_builders = kwargs['target_builders']
        if 'strict_validation' in kwargs:
            config.strict_validation = kwargs['strict_validation']
        
        start_time = datetime.now()
        logger.info(f"Starting unified validation at: {self.workspace_root}")
        
        try:
            # Detect workspaces using unified approach
            detected_workspaces = self.workspace_detector.detect_workspaces()
            workspace_type = self.workspace_detector.get_workspace_type()
            
            if not detected_workspaces:
                logger.warning("No workspaces detected for validation")
                return create_empty_result(
                    workspace_root=str(self.workspace_root),
                    workspace_type=workspace_type,
                    error="No valid workspaces found for validation",
                    start_time=start_time
                )
            
            logger.info(f"Detected {len(detected_workspaces)} workspace(s) "
                       f"of type '{workspace_type}' for validation")
            
            # Create result builder
            result_builder = ValidationResultBuilder(
                workspace_root=str(self.workspace_root),
                workspace_type=workspace_type,
                start_time=start_time
            )
            
            # Validate each workspace entry using unified logic
            for workspace_id, workspace_info in detected_workspaces.items():
                logger.info(f"Validating workspace: {workspace_id}")
                
                workspace_result = self.validate_single_workspace_entry(
                    workspace_id=workspace_id,
                    workspace_info=workspace_info,
                    config=config
                )
                
                result_builder.add_workspace_result(
                    workspace_id=workspace_result.workspace_id,
                    workspace_type=workspace_result.workspace_type,
                    workspace_path=workspace_result.workspace_path,
                    success=workspace_result.success,
                    alignment_results=workspace_result.alignment_results,
                    builder_results=workspace_result.builder_results,
                    error=workspace_result.error,
                    warnings=workspace_result.warnings,
                    developer_info=workspace_result.developer_info,
                    validation_start_time=datetime.fromisoformat(workspace_result.validation_start_time) if workspace_result.validation_start_time else None,
                    validation_end_time=datetime.fromisoformat(workspace_result.validation_end_time) if workspace_result.validation_end_time else None
                )
            
            # Generate recommendations based on results
            recommendations = self._generate_unified_recommendations(
                detected_workspaces, result_builder.workspaces
            )
            for rec in recommendations:
                result_builder.add_recommendation(rec)
            
            # Build final result
            final_result = result_builder.build()
            
            logger.info(f"Completed unified validation: "
                       f"{final_result.summary.successful_workspaces}/"
                       f"{final_result.summary.total_workspaces} successful")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Unified validation failed: {e}")
            
            # Create error result
            result_builder = ValidationResultBuilder(
                workspace_root=str(self.workspace_root),
                workspace_type="unknown",
                start_time=start_time
            )
            
            result_builder.set_global_error(str(e))
            result_builder.add_recommendation("Fix validation setup issues before retrying")
            
            return result_builder.build()
    
    def validate_single_workspace_entry(
        self,
        workspace_id: str,
        workspace_info: Dict[str, Any],
        config: ValidationConfig
    ) -> WorkspaceValidationResult:
        """
        Validate one workspace entry (used by both single and multi scenarios).
        
        This method provides the core validation logic that is applied
        consistently to each workspace, regardless of whether it's part
        of a single or multi-workspace scenario.
        
        Args:
            workspace_id: Identifier for the workspace
            workspace_info: Information about the workspace
            config: Validation configuration
            
        Returns:
            WorkspaceValidationResult for this specific workspace
        """
        workspace_start_time = datetime.now()
        
        try:
            logger.info(f"Starting validation for workspace '{workspace_id}'")
            
            # Extract workspace information
            workspace_type = workspace_info.get("workspace_type", "unknown")
            workspace_path = workspace_info.get("workspace_path", "")
            developer_info = workspace_info.get("developer_info", {})
            
            # Initialize validation results
            alignment_results = None
            builder_results = None
            success = True
            error = None
            warnings = []
            
            # Run alignment validation if requested
            if 'alignment' in config.validation_types or 'all' in config.validation_types:
                try:
                    alignment_results = self._run_workspace_alignment_validation(
                        workspace_id, workspace_info, config
                    )
                    
                    # Check for alignment failures
                    if alignment_results and not alignment_results.get('success', True):
                        success = False
                        
                except Exception as e:
                    logger.error(f"Alignment validation failed for '{workspace_id}': {e}")
                    success = False
                    warnings.append(f"Alignment validation error: {str(e)}")
                    alignment_results = {
                        'success': False,
                        'error': str(e),
                        'workspace_id': workspace_id
                    }
            
            # Run builder validation if requested
            if 'builders' in config.validation_types or 'all' in config.validation_types:
                try:
                    builder_results = self._run_workspace_builder_validation(
                        workspace_id, workspace_info, config
                    )
                    
                    # Check for builder failures
                    if builder_results and not builder_results.get('success', True):
                        success = False
                        
                except Exception as e:
                    logger.error(f"Builder validation failed for '{workspace_id}': {e}")
                    success = False
                    warnings.append(f"Builder validation error: {str(e)}")
                    builder_results = {
                        'success': False,
                        'error': str(e),
                        'workspace_id': workspace_id
                    }
            
            workspace_end_time = datetime.now()
            
            # Create workspace validation result
            workspace_result = WorkspaceValidationResult(
                workspace_id=workspace_id,
                workspace_type=workspace_type,
                workspace_path=workspace_path,
                success=success,
                validation_start_time=workspace_start_time.isoformat(),
                validation_end_time=workspace_end_time.isoformat(),
                validation_duration_seconds=(workspace_end_time - workspace_start_time).total_seconds(),
                alignment_results=alignment_results,
                builder_results=builder_results,
                error=error,
                warnings=warnings,
                developer_info=developer_info
            )
            
            logger.info(f"Completed validation for workspace '{workspace_id}': "
                       f"{'SUCCESS' if success else 'FAILED'}")
            
            return workspace_result
            
        except Exception as e:
            logger.error(f"Workspace validation failed for '{workspace_id}': {e}")
            
            workspace_end_time = datetime.now()
            
            return WorkspaceValidationResult(
                workspace_id=workspace_id,
                workspace_type=workspace_info.get("workspace_type", "unknown"),
                workspace_path=workspace_info.get("workspace_path", ""),
                success=False,
                validation_start_time=workspace_start_time.isoformat(),
                validation_end_time=workspace_end_time.isoformat(),
                validation_duration_seconds=(workspace_end_time - workspace_start_time).total_seconds(),
                error=str(e),
                warnings=[],
                developer_info=workspace_info.get("developer_info", {})
            )
    
    def _run_workspace_alignment_validation(
        self,
        workspace_id: str,
        workspace_info: Dict[str, Any],
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Run alignment validation for a workspace."""
        try:
            # Determine developer ID for alignment tester
            if workspace_id == "default":
                # Single workspace scenario
                developer_id = None
            elif workspace_id == "shared":
                # Shared workspace
                developer_id = "shared"
            else:
                # Multi-workspace scenario
                developer_id = workspace_id
            
            # Create alignment tester
            alignment_tester = WorkspaceUnifiedAlignmentTester(
                workspace_root=self.workspace_root,
                developer_id=developer_id
            )
            
            # Run workspace validation
            alignment_results = alignment_tester.run_workspace_validation(
                target_scripts=config.target_scripts,
                skip_levels=config.skip_levels,
                workspace_context=config.workspace_context
            )
            
            return alignment_results
            
        except Exception as e:
            logger.error(f"Failed to run alignment validation for '{workspace_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'workspace_id': workspace_id,
                'validation_type': 'alignment'
            }
    
    def _run_workspace_builder_validation(
        self,
        workspace_id: str,
        workspace_info: Dict[str, Any],
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Run builder validation for a workspace."""
        try:
            # Determine developer ID for builder tester
            if workspace_id == "default":
                # Single workspace scenario
                developer_id = None
            elif workspace_id == "shared":
                # Shared workspace
                developer_id = "shared"
            else:
                # Multi-workspace scenario
                developer_id = workspace_id
            
            # Create builder tester
            builder_tester = WorkspaceUniversalStepBuilderTest(
                workspace_root=self.workspace_root,
                developer_id=developer_id,
                builder_file_path=""  # Will be determined by tester
            )
            
            # Run workspace builder test
            builder_results = builder_tester.run_workspace_builder_test()
            
            # Filter results if specific builders were requested
            if config.target_builders and 'results' in builder_results:
                filtered_results = {
                    builder_name: result
                    for builder_name, result in builder_results['results'].items()
                    if builder_name in config.target_builders
                }
                builder_results['results'] = filtered_results
                builder_results['tested_builders'] = len(filtered_results)
                
                # Recalculate success counts
                successful_tests = sum(
                    1 for result in filtered_results.values()
                    if result.get('success', False)
                )
                builder_results['successful_tests'] = successful_tests
                builder_results['failed_tests'] = len(filtered_results) - successful_tests
            
            return builder_results
            
        except Exception as e:
            logger.error(f"Failed to run builder validation for '{workspace_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'workspace_id': workspace_id,
                'validation_type': 'builders'
            }
    
    def _generate_unified_recommendations(
        self,
        detected_workspaces: Dict[str, Any],
        validation_results: Dict[str, WorkspaceValidationResult]
    ) -> List[str]:
        """Generate recommendations based on unified validation results."""
        recommendations = []
        
        try:
            # Calculate success statistics
            total_workspaces = len(validation_results)
            successful_workspaces = sum(
                1 for result in validation_results.values() if result.success
            )
            success_rate = successful_workspaces / total_workspaces if total_workspaces > 0 else 0.0
            
            # Recommendations based on success rate
            if success_rate < 0.5:
                recommendations.append(
                    f"Low success rate ({success_rate:.1%}). "
                    "Review workspace setup and validation configuration."
                )
            elif success_rate < 0.8:
                recommendations.append(
                    f"Moderate success rate ({success_rate:.1%}). "
                    "Address common issues to improve workspace validation."
                )
            elif success_rate == 1.0:
                recommendations.append(
                    "All workspaces passed validation successfully. "
                    "Consider adding more comprehensive validation tests."
                )
            else:
                recommendations.append(
                    f"Good success rate ({success_rate:.1%}). "
                    "Address remaining issues for complete validation success."
                )
            
            # Recommendations for failed workspaces
            failed_workspaces = [
                workspace_id for workspace_id, result in validation_results.items()
                if not result.success
            ]
            
            if failed_workspaces:
                if len(failed_workspaces) == 1:
                    recommendations.append(
                        f"Review and fix validation issues in workspace: {failed_workspaces[0]}"
                    )
                else:
                    recommendations.append(
                        f"Review and fix validation issues in workspaces: {', '.join(failed_workspaces)}"
                    )
            
            # Workspace-specific recommendations
            if total_workspaces == 1:
                workspace_type = self.workspace_detector.get_workspace_type()
                if workspace_type == "single":
                    recommendations.append(
                        "Consider creating additional developer workspaces to test "
                        "multi-workspace scenarios and improve development isolation."
                    )
            elif total_workspaces > 10:
                recommendations.append(
                    "Large number of workspaces detected. "
                    "Consider implementing workspace grouping or batch validation strategies."
                )
            
            # Validation type recommendations
            alignment_failures = sum(
                1 for result in validation_results.values()
                if result.alignment_results and not result.alignment_results.get('success', True)
            )
            
            builder_failures = sum(
                1 for result in validation_results.values()
                if result.builder_results and not result.builder_results.get('success', True)
            )
            
            if alignment_failures > 0:
                recommendations.append(
                    f"Address alignment validation issues in {alignment_failures} workspace(s). "
                    "Check script-contract-spec-builder alignment."
                )
            
            if builder_failures > 0:
                recommendations.append(
                    f"Address builder validation issues in {builder_failures} workspace(s). "
                    "Check builder implementation and configuration."
                )
            
            # Default recommendation if none generated
            if not recommendations:
                recommendations.append(
                    "Workspace validation completed. "
                    "Continue developing and testing workspace-specific components."
                )
        
        except Exception as e:
            logger.warning(f"Failed to generate unified recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error.")
        
        return recommendations
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of detected workspace structure."""
        return self.workspace_detector.get_summary()
    
    def validate_workspace_structure(self) -> tuple[bool, List[str]]:
        """Validate the workspace structure before running validation."""
        return self.workspace_detector.validate_workspace_structure()
