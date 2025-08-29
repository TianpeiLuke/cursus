"""
Workspace Validation Orchestrator

High-level orchestrator for workspace validation operations.
Coordinates alignment and builder validation across multiple workspaces
with comprehensive reporting and error handling.

Architecture:
- Coordinates WorkspaceUnifiedAlignmentTester and WorkspaceUniversalStepBuilderTest
- Provides unified validation interface for single and multi-workspace scenarios
- Supports parallel validation for performance optimization
- Generates comprehensive validation reports with workspace context

Features:
- Single workspace comprehensive validation
- Multi-workspace validation coordination
- Parallel validation support for performance
- Detailed validation reporting and diagnostics
- Cross-workspace dependency analysis
- Validation result aggregation and summarization
"""

import os
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
import logging
from datetime import datetime

from .workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
from .workspace_builder_test import WorkspaceUniversalStepBuilderTest
from .workspace_manager import WorkspaceManager


logger = logging.getLogger(__name__)


class WorkspaceValidationOrchestrator:
    """
    High-level orchestrator for workspace validation operations.
    
    Coordinates comprehensive validation across multiple workspaces including:
    - Alignment validation across all 4 levels
    - Builder testing and validation
    - Cross-workspace dependency analysis
    - Comprehensive reporting and diagnostics
    
    Features:
    - Single and multi-workspace validation
    - Parallel validation for performance
    - Detailed error reporting and recommendations
    - Validation result aggregation and analysis
    """
    
    def __init__(
        self,
        workspace_root: Union[str, Path],
        enable_parallel_validation: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize workspace validation orchestrator.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            enable_parallel_validation: Whether to enable parallel validation
            max_workers: Maximum number of parallel workers (None for auto)
        """
        self.workspace_root = Path(workspace_root)
        self.enable_parallel_validation = enable_parallel_validation
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
        
        # Initialize workspace manager
        self.workspace_manager = WorkspaceManager(workspace_root=workspace_root)
        
        logger.info(f"Initialized workspace validation orchestrator at '{workspace_root}' "
                   f"with parallel validation {'enabled' if enable_parallel_validation else 'disabled'}")
    
    def validate_workspace(
        self,
        developer_id: str,
        validation_levels: Optional[List[str]] = None,
        target_scripts: Optional[List[str]] = None,
        target_builders: Optional[List[str]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation for a single workspace.
        
        Args:
            developer_id: Developer workspace to validate
            validation_levels: Validation types to run ('alignment', 'builders', 'all')
            target_scripts: Specific scripts to validate (None for all)
            target_builders: Specific builders to validate (None for all)
            validation_config: Additional validation configuration
            
        Returns:
            Comprehensive validation results for the workspace
        """
        logger.info(f"Starting comprehensive validation for developer '{developer_id}'")
        
        # Default validation levels
        if validation_levels is None:
            validation_levels = ['alignment', 'builders']
        
        # Default validation config
        if validation_config is None:
            validation_config = {}
        
        validation_start_time = datetime.now()
        
        try:
            # Validate developer exists
            available_developers = self.workspace_manager.list_available_developers()
            if developer_id not in available_developers:
                raise ValueError(f"Developer workspace not found: {developer_id}")
            
            # Initialize validation results
            validation_results = {
                'developer_id': developer_id,
                'workspace_root': str(self.workspace_root),
                'validation_start_time': validation_start_time.isoformat(),
                'validation_levels': validation_levels,
                'success': True,
                'results': {},
                'summary': {},
                'recommendations': []
            }
            
            # Run alignment validation if requested
            if 'alignment' in validation_levels or 'all' in validation_levels:
                logger.info(f"Running alignment validation for developer '{developer_id}'")
                alignment_results = self._run_alignment_validation(
                    developer_id, target_scripts, validation_config
                )
                validation_results['results']['alignment'] = alignment_results
                
                if not alignment_results.get('success', False):
                    validation_results['success'] = False
            
            # Run builder validation if requested
            if 'builders' in validation_levels or 'all' in validation_levels:
                logger.info(f"Running builder validation for developer '{developer_id}'")
                builder_results = self._run_builder_validation(
                    developer_id, target_builders, validation_config
                )
                validation_results['results']['builders'] = builder_results
                
                if not builder_results.get('success', False):
                    validation_results['success'] = False
            
            # Generate validation summary
            validation_results['summary'] = self._generate_validation_summary(
                validation_results['results']
            )
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_validation_recommendations(
                validation_results['results']
            )
            
            # Calculate validation duration
            validation_end_time = datetime.now()
            validation_results['validation_end_time'] = validation_end_time.isoformat()
            validation_results['validation_duration_seconds'] = (
                validation_end_time - validation_start_time
            ).total_seconds()
            
            logger.info(f"Completed comprehensive validation for developer '{developer_id}': "
                       f"{'SUCCESS' if validation_results['success'] else 'FAILED'}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed for developer '{developer_id}': {e}")
            validation_end_time = datetime.now()
            
            return {
                'developer_id': developer_id,
                'workspace_root': str(self.workspace_root),
                'validation_start_time': validation_start_time.isoformat(),
                'validation_end_time': validation_end_time.isoformat(),
                'validation_duration_seconds': (validation_end_time - validation_start_time).total_seconds(),
                'validation_levels': validation_levels,
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
        Run validation across all discovered workspaces.
        
        Args:
            validation_levels: Validation types to run ('alignment', 'builders', 'all')
            parallel: Whether to run validations in parallel (None for default)
            validation_config: Additional validation configuration
            
        Returns:
            Aggregated validation results for all workspaces
        """
        logger.info("Starting validation across all workspaces")
        
        # Use instance default for parallel if not specified
        if parallel is None:
            parallel = self.enable_parallel_validation
        
        validation_start_time = datetime.now()
        
        try:
            # Discover all available developers
            available_developers = self.workspace_manager.list_available_developers()
            
            if not available_developers:
                logger.warning("No developer workspaces found")
                return {
                    'workspace_root': str(self.workspace_root),
                    'validation_start_time': validation_start_time.isoformat(),
                    'validation_end_time': datetime.now().isoformat(),
                    'total_workspaces': 0,
                    'validated_workspaces': 0,
                    'successful_validations': 0,
                    'failed_validations': 0,
                    'success': True,
                    'results': {},
                    'summary': {'message': 'No workspaces found to validate'},
                    'recommendations': ['Create developer workspaces to enable validation']
                }
            
            logger.info(f"Found {len(available_developers)} developer workspaces: {available_developers}")
            
            # Run validations
            if parallel and len(available_developers) > 1:
                all_results = self._run_parallel_validations(
                    available_developers, validation_levels, validation_config
                )
            else:
                all_results = self._run_sequential_validations(
                    available_developers, validation_levels, validation_config
                )
            
            # Aggregate results
            aggregated_results = self._aggregate_validation_results(
                all_results, validation_start_time
            )
            
            logger.info(f"Completed validation across all workspaces: "
                       f"{aggregated_results['successful_validations']}/{aggregated_results['total_workspaces']} successful")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Multi-workspace validation failed: {e}")
            validation_end_time = datetime.now()
            
            return {
                'workspace_root': str(self.workspace_root),
                'validation_start_time': validation_start_time.isoformat(),
                'validation_end_time': validation_end_time.isoformat(),
                'validation_duration_seconds': (validation_end_time - validation_start_time).total_seconds(),
                'total_workspaces': 0,
                'validated_workspaces': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'success': False,
                'error': str(e),
                'results': {},
                'summary': {'error': 'Multi-workspace validation failed to complete'},
                'recommendations': ['Fix validation setup issues before retrying']
            }
    
    def _run_alignment_validation(
        self,
        developer_id: str,
        target_scripts: Optional[List[str]],
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run alignment validation for a specific workspace."""
        try:
            # Create alignment tester
            alignment_tester = WorkspaceUnifiedAlignmentTester(
                workspace_root=self.workspace_root,
                developer_id=developer_id,
                **validation_config.get('alignment', {})
            )
            
            # Run workspace validation
            alignment_results = alignment_tester.run_workspace_validation(
                target_scripts=target_scripts,
                skip_levels=validation_config.get('skip_levels'),
                workspace_context=validation_config.get('workspace_context')
            )
            
            return alignment_results
            
        except Exception as e:
            logger.error(f"Alignment validation failed for developer '{developer_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'developer_id': developer_id,
                'validation_type': 'alignment'
            }
    
    def _run_builder_validation(
        self,
        developer_id: str,
        target_builders: Optional[List[str]],
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run builder validation for a specific workspace."""
        try:
            # Run validation for all builders in workspace
            builder_results = WorkspaceUniversalStepBuilderTest.test_all_workspace_builders(
                workspace_root=self.workspace_root,
                developer_id=developer_id,
                test_config=validation_config.get('builder_test_config'),
                **validation_config.get('builders', {})
            )
            
            # Filter results if specific builders were requested
            if target_builders and 'results' in builder_results:
                filtered_results = {
                    builder_name: result
                    for builder_name, result in builder_results['results'].items()
                    if builder_name in target_builders
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
            logger.error(f"Builder validation failed for developer '{developer_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'developer_id': developer_id,
                'validation_type': 'builders'
            }
    
    def _run_parallel_validations(
        self,
        developers: List[str],
        validation_levels: Optional[List[str]],
        validation_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run validations in parallel across multiple workspaces."""
        logger.info(f"Running parallel validation for {len(developers)} workspaces")
        
        all_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit validation tasks
            future_to_developer = {
                executor.submit(
                    self.validate_workspace,
                    developer_id,
                    validation_levels,
                    None,  # target_scripts
                    None,  # target_builders
                    validation_config
                ): developer_id
                for developer_id in developers
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_developer):
                developer_id = future_to_developer[future]
                try:
                    result = future.result()
                    all_results[developer_id] = result
                    logger.info(f"Completed validation for developer '{developer_id}': "
                               f"{'SUCCESS' if result.get('success', False) else 'FAILED'}")
                except Exception as e:
                    logger.error(f"Parallel validation failed for developer '{developer_id}': {e}")
                    all_results[developer_id] = {
                        'success': False,
                        'error': str(e),
                        'developer_id': developer_id
                    }
        
        return all_results
    
    def _run_sequential_validations(
        self,
        developers: List[str],
        validation_levels: Optional[List[str]],
        validation_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run validations sequentially across multiple workspaces."""
        logger.info(f"Running sequential validation for {len(developers)} workspaces")
        
        all_results = {}
        
        for developer_id in developers:
            logger.info(f"Validating workspace for developer '{developer_id}'")
            
            result = self.validate_workspace(
                developer_id=developer_id,
                validation_levels=validation_levels,
                validation_config=validation_config
            )
            
            all_results[developer_id] = result
            logger.info(f"Completed validation for developer '{developer_id}': "
                       f"{'SUCCESS' if result.get('success', False) else 'FAILED'}")
        
        return all_results
    
    def _aggregate_validation_results(
        self,
        all_results: Dict[str, Dict[str, Any]],
        validation_start_time: datetime
    ) -> Dict[str, Any]:
        """Aggregate validation results from multiple workspaces."""
        validation_end_time = datetime.now()
        
        # Calculate basic statistics
        total_workspaces = len(all_results)
        successful_validations = sum(
            1 for result in all_results.values()
            if result.get('success', False)
        )
        failed_validations = total_workspaces - successful_validations
        
        # Aggregate detailed results
        aggregated_results = {
            'workspace_root': str(self.workspace_root),
            'validation_start_time': validation_start_time.isoformat(),
            'validation_end_time': validation_end_time.isoformat(),
            'validation_duration_seconds': (validation_end_time - validation_start_time).total_seconds(),
            'total_workspaces': total_workspaces,
            'validated_workspaces': total_workspaces,
            'successful_validations': successful_validations,
            'failed_validations': failed_validations,
            'success_rate': successful_validations / total_workspaces if total_workspaces > 0 else 0.0,
            'success': failed_validations == 0,
            'results': all_results,
            'summary': self._generate_multi_workspace_summary(all_results),
            'recommendations': self._generate_multi_workspace_recommendations(all_results)
        }
        
        return aggregated_results
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for single workspace validation."""
        summary = {
            'validation_types_run': list(results.keys()),
            'overall_success': all(result.get('success', False) for result in results.values()),
            'details': {}
        }
        
        # Summarize alignment results
        if 'alignment' in results:
            alignment_result = results['alignment']
            summary['details']['alignment'] = {
                'success': alignment_result.get('success', False),
                'scripts_validated': len(alignment_result.get('results', {})),
                'cross_workspace_validation': 'cross_workspace_validation' in alignment_result
            }
        
        # Summarize builder results
        if 'builders' in results:
            builder_result = results['builders']
            summary['details']['builders'] = {
                'success': builder_result.get('success', False),
                'total_builders': builder_result.get('total_builders', 0),
                'successful_tests': builder_result.get('successful_tests', 0),
                'failed_tests': builder_result.get('failed_tests', 0)
            }
        
        return summary
    
    def _generate_validation_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for single workspace validation."""
        recommendations = []
        
        try:
            # Alignment recommendations
            if 'alignment' in results:
                alignment_result = results['alignment']
                if 'cross_workspace_validation' in alignment_result:
                    cross_workspace = alignment_result['cross_workspace_validation']
                    if 'recommendations' in cross_workspace:
                        recommendations.extend(cross_workspace['recommendations'])
            
            # Builder recommendations
            if 'builders' in results:
                builder_result = results['builders']
                if 'summary' in builder_result and 'recommendations' in builder_result['summary']:
                    recommendations.extend(builder_result['summary']['recommendations'])
            
            # General recommendations
            if not recommendations:
                recommendations.append("Workspace validation completed successfully. "
                                     "Consider adding more workspace-specific components for better isolation.")
        
        except Exception as e:
            logger.warning(f"Failed to generate validation recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error.")
        
        return recommendations
    
    def _generate_multi_workspace_summary(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary for multi-workspace validation."""
        summary = {
            'overall_statistics': {
                'total_workspaces': len(all_results),
                'successful_workspaces': 0,
                'failed_workspaces': 0,
                'success_rate': 0.0
            },
            'validation_type_statistics': {},
            'common_issues': [],
            'workspace_details': {}
        }
        
        try:
            # Calculate overall statistics
            successful_workspaces = sum(
                1 for result in all_results.values()
                if result.get('success', False)
            )
            summary['overall_statistics']['successful_workspaces'] = successful_workspaces
            summary['overall_statistics']['failed_workspaces'] = len(all_results) - successful_workspaces
            summary['overall_statistics']['success_rate'] = (
                successful_workspaces / len(all_results) if all_results else 0.0
            )
            
            # Analyze validation type statistics
            validation_types = set()
            for result in all_results.values():
                if 'results' in result:
                    validation_types.update(result['results'].keys())
            
            for validation_type in validation_types:
                type_stats = {
                    'workspaces_run': 0,
                    'successful': 0,
                    'failed': 0,
                    'success_rate': 0.0
                }
                
                for result in all_results.values():
                    if 'results' in result and validation_type in result['results']:
                        type_stats['workspaces_run'] += 1
                        if result['results'][validation_type].get('success', False):
                            type_stats['successful'] += 1
                        else:
                            type_stats['failed'] += 1
                
                if type_stats['workspaces_run'] > 0:
                    type_stats['success_rate'] = type_stats['successful'] / type_stats['workspaces_run']
                
                summary['validation_type_statistics'][validation_type] = type_stats
            
            # Analyze common issues across workspaces
            all_issues = []
            for developer_id, result in all_results.items():
                if not result.get('success', False) and 'error' in result:
                    all_issues.append({
                        'workspace': developer_id,
                        'type': 'validation_error',
                        'description': result['error']
                    })
            
            # Group similar issues
            issue_counts = {}
            for issue in all_issues:
                issue_type = issue['type']
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            # Identify common issues (appearing in >25% of workspaces)
            threshold = len(all_results) * 0.25
            common_issues = [
                {'type': issue_type, 'count': count, 'percentage': count / len(all_results)}
                for issue_type, count in issue_counts.items()
                if count > threshold
            ]
            summary['common_issues'] = common_issues
            
            # Generate workspace details
            for developer_id, result in all_results.items():
                summary['workspace_details'][developer_id] = {
                    'success': result.get('success', False),
                    'validation_types': list(result.get('results', {}).keys()),
                    'has_error': 'error' in result
                }
        
        except Exception as e:
            logger.warning(f"Failed to generate multi-workspace summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def _generate_multi_workspace_recommendations(self, all_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations for multi-workspace validation."""
        recommendations = []
        
        try:
            # Calculate success rate
            successful_workspaces = sum(
                1 for result in all_results.values()
                if result.get('success', False)
            )
            success_rate = successful_workspaces / len(all_results) if all_results else 0.0
            
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
            else:
                recommendations.append(
                    f"Good success rate ({success_rate:.1%}). "
                    "Consider standardizing successful patterns across all workspaces."
                )
            
            # Recommendations for failed workspaces
            failed_workspaces = [
                developer_id for developer_id, result in all_results.items()
                if not result.get('success', False)
            ]
            
            if failed_workspaces:
                recommendations.append(
                    f"Review and fix validation issues in workspaces: {', '.join(failed_workspaces)}"
                )
            
            # General recommendations
            if len(all_results) == 1:
                recommendations.append(
                    "Consider creating additional developer workspaces to test multi-workspace scenarios."
                )
            elif len(all_results) > 10:
                recommendations.append(
                    "Large number of workspaces detected. "
                    "Consider implementing workspace grouping or batch validation strategies."
                )
        
        except Exception as e:
            logger.warning(f"Failed to generate multi-workspace recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error.")
        
        return recommendations
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about orchestrator configuration."""
        return {
            'workspace_root': str(self.workspace_root),
            'enable_parallel_validation': self.enable_parallel_validation,
            'max_workers': self.max_workers,
            'workspace_manager_info': self.workspace_manager.get_workspace_info().model_dump(),
            'available_developers': self.workspace_manager.list_available_developers()
        }
    
    @classmethod
    def create_from_workspace_manager(
        cls,
        workspace_manager: WorkspaceManager,
        **kwargs
    ) -> 'WorkspaceValidationOrchestrator':
        """Create orchestrator from existing WorkspaceManager."""
        return cls(
            workspace_root=workspace_manager.workspace_root,
            **kwargs
        )
