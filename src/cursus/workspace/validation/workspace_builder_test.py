"""
Workspace Universal Step Builder Test

Extends UniversalStepBuilderTest to support multi-developer workspace structures.
Provides workspace-aware builder testing while maintaining full API compatibility
with the existing UniversalStepBuilderTest.

Architecture:
- Extends existing UniversalStepBuilderTest capabilities
- Supports workspace-specific builder discovery and testing
- Maintains backward compatibility with existing testing workflows
- Provides workspace-specific reporting and diagnostics

Workspace Integration:
- Uses WorkspaceModuleLoader for dynamic builder class loading
- Integrates with DeveloperWorkspaceFileResolver for builder discovery
- Supports testing builders from multiple workspaces
- Provides workspace-specific error reporting and diagnostics
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Type, Tuple
import logging

from ...validation.builders.universal_test import UniversalStepBuilderTest
from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_module_loader import WorkspaceModuleLoader
from .workspace_manager import WorkspaceManager


logger = logging.getLogger(__name__)


class WorkspaceUniversalStepBuilderTest(UniversalStepBuilderTest):
    """
    Workspace-aware version of UniversalStepBuilderTest.
    
    Extends the existing UniversalStepBuilderTest to support multi-developer
    workspace structures while maintaining full API compatibility.
    
    Features:
    - Workspace-aware builder discovery and testing
    - Dynamic loading of builder classes from workspace directories
    - Workspace-specific error reporting and diagnostics
    - Full backward compatibility with UniversalStepBuilderTest
    - Support for testing builders across multiple workspaces
    """
    
    def __init__(
        self,
        workspace_root: Union[str, Path],
        developer_id: str,
        builder_file_path: str,
        enable_shared_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize workspace-aware step builder test.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            developer_id: Specific developer workspace to target
            builder_file_path: Path to the builder file to test
            enable_shared_fallback: Whether to fallback to shared workspace
            **kwargs: Additional arguments passed to UniversalStepBuilderTest
        """
        self.workspace_root = Path(workspace_root)
        self.developer_id = developer_id
        self.enable_shared_fallback = enable_shared_fallback
        self.builder_file_path = builder_file_path
        
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
        
        # Load builder class from workspace
        try:
            self.builder_class = self._load_workspace_builder_class(builder_file_path)
        except Exception as e:
            logger.warning(f"Failed to load builder class from workspace: {e}")
            self.builder_class = None
        
        # Initialize parent with workspace-loaded builder (if available)
        if self.builder_class:
            super().__init__(
                builder_class=self.builder_class,
                **kwargs
            )
        else:
            # Initialize with mock builder class for testing
            mock_builder_class = type('MockBuilder', (), {
                '__name__': 'MockBuilder',
                '__module__': 'mock_module'
            })
            super().__init__(
                builder_class=mock_builder_class,
                **kwargs
            )
        
        logger.info(f"Initialized workspace builder test for developer '{developer_id}' "
                   f"with builder '{builder_file_path}'")
    
    def _load_workspace_builder_class(self, builder_file_path: str) -> Type:
        """Load builder class from workspace using WorkspaceModuleLoader."""
        try:
            # Extract step name from builder file path
            builder_file = Path(builder_file_path)
            step_name = self._extract_step_name_from_builder_file(builder_file)
            
            # Load builder class using workspace module loader
            builder_class = self.module_loader.load_builder_class(step_name)
            
            if builder_class is None:
                raise ValueError(f"Failed to load builder class for step '{step_name}' "
                               f"from file '{builder_file_path}'")
            
            logger.info(f"Successfully loaded builder class: {builder_class.__name__}")
            return builder_class
            
        except Exception as e:
            logger.error(f"Failed to load workspace builder class from '{builder_file_path}': {e}")
            raise
    
    def _extract_step_name_from_builder_file(self, builder_file: Path) -> str:
        """Extract step name from builder file path."""
        # Handle different builder file naming patterns
        filename = builder_file.stem
        
        # Pattern: builder_<step_name>_step.py
        if filename.startswith('builder_') and filename.endswith('_step'):
            return filename[8:-5]  # Remove 'builder_' prefix and '_step' suffix
        
        # Pattern: <step_name>_builder.py
        if filename.endswith('_builder'):
            return filename[:-8]  # Remove '_builder' suffix
        
        # Pattern: builder_<step_name>.py
        if filename.startswith('builder_'):
            return filename[8:]  # Remove 'builder_' prefix
        
        # Fallback: use filename as-is
        logger.warning(f"Could not extract step name from builder file '{filename}', using as-is")
        return filename
    
    def run_workspace_builder_test(
        self,
        test_config: Optional[Dict[str, Any]] = None,
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run builder test with workspace-specific context.
        
        Args:
            test_config: Test configuration parameters
            workspace_context: Additional workspace context for testing
            
        Returns:
            Comprehensive test results with workspace context
        """
        logger.info(f"Starting workspace builder test for developer '{self.developer_id}'")
        
        try:
            # Run standard builder test
            test_results = self.run_test(test_config=test_config)
            
            # Enhance results with workspace-specific information
            workspace_results = self._enhance_results_with_workspace_context(
                test_results, workspace_context
            )
            
            # Add workspace-specific validation
            workspace_validation = self._validate_workspace_builder_integration()
            workspace_results['workspace_validation'] = workspace_validation
            
            logger.info(f"Completed workspace builder test for developer '{self.developer_id}'")
            return workspace_results
            
        except Exception as e:
            logger.error(f"Workspace builder test failed for developer '{self.developer_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'workspace_context': {
                    'developer_id': self.developer_id,
                    'workspace_root': str(self.workspace_root),
                    'builder_file_path': self.builder_file_path,
                    'enable_shared_fallback': self.enable_shared_fallback
                }
            }
    
    def _enhance_results_with_workspace_context(
        self,
        test_results: Dict[str, Any],
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance test results with workspace-specific context."""
        enhanced_results = test_results.copy()
        
        # Add workspace metadata
        enhanced_results['workspace_metadata'] = {
            'developer_id': self.developer_id,
            'workspace_root': str(self.workspace_root),
            'builder_file_path': self.builder_file_path,
            'enable_shared_fallback': self.enable_shared_fallback,
            'builder_class_name': self.builder_class.__name__ if self.builder_class else None,
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
            test_results
        )
        
        return enhanced_results
    
    def _add_workspace_context_to_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add workspace context to error messages."""
        enhanced_errors = []
        
        for error in errors:
            enhanced_error = error.copy()
            enhanced_error['workspace_context'] = {
                'developer_id': self.developer_id,
                'workspace_root': str(self.workspace_root),
                'builder_file_path': self.builder_file_path
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
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workspace-specific test statistics."""
        stats = {
            'builder_loaded_from_workspace': True,
            'builder_class_name': self.builder_class.__name__ if self.builder_class else None,
            'builder_module_path': getattr(self.builder_class, '__module__', None) if self.builder_class else None,
            'workspace_components_available': {},
            'shared_fallback_used': False
        }
        
        # Check workspace component availability
        try:
            step_name = self._extract_step_name_from_builder_file(Path(self.builder_file_path))
            
            # Check each component type
            for component_type in ['contracts', 'specs', 'configs']:
                method_name = f'find_{component_type.rstrip("s")}_file'
                if hasattr(self.file_resolver, method_name):
                    method = getattr(self.file_resolver, method_name)
                    component_path = method(step_name)
                    
                    stats['workspace_components_available'][component_type] = component_path is not None
                    
                    # Check if shared fallback was used
                    if component_path and 'shared' in component_path:
                        stats['shared_fallback_used'] = True
        
        except Exception as e:
            logger.warning(f"Failed to generate workspace statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _validate_workspace_builder_integration(self) -> Dict[str, Any]:
        """Validate workspace builder integration and dependencies."""
        validation_results = {
            'builder_class_valid': False,
            'workspace_dependencies_available': {},
            'integration_issues': [],
            'recommendations': []
        }
        
        try:
            # Validate builder class
            if self.builder_class:
                validation_results['builder_class_valid'] = True
                validation_results['builder_class_info'] = {
                    'name': self.builder_class.__name__,
                    'module': getattr(self.builder_class, '__module__', None),
                    'file': getattr(self.builder_class, '__file__', None)
                }
            
            # Check workspace dependencies
            step_name = self._extract_step_name_from_builder_file(Path(self.builder_file_path))
            
            for component_type in ['contracts', 'specs', 'configs']:
                method_name = f'find_{component_type.rstrip("s")}_file'
                if hasattr(self.file_resolver, method_name):
                    method = getattr(self.file_resolver, method_name)
                    component_path = method(step_name)
                    
                    validation_results['workspace_dependencies_available'][component_type] = {
                        'available': component_path is not None,
                        'path': component_path,
                        'from_shared': component_path and 'shared' in component_path if component_path else False
                    }
            
            # Check for integration issues
            issues = self._detect_integration_issues(validation_results)
            validation_results['integration_issues'] = issues
            
            # Generate recommendations
            recommendations = self._generate_integration_recommendations(validation_results)
            validation_results['recommendations'] = recommendations
            
        except Exception as e:
            logger.warning(f"Workspace builder integration validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _detect_integration_issues(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential integration issues with workspace builder."""
        issues = []
        
        try:
            # Check for missing dependencies
            dependencies = validation_results.get('workspace_dependencies_available', {})
            missing_deps = [
                dep_type for dep_type, dep_info in dependencies.items()
                if not dep_info.get('available', False)
            ]
            
            if missing_deps:
                issues.append({
                    'type': 'missing_dependencies',
                    'severity': 'warning',
                    'description': f'Missing workspace dependencies: {", ".join(missing_deps)}',
                    'missing_dependencies': missing_deps
                })
            
            # Check for mixed workspace/shared usage
            shared_deps = [
                dep_type for dep_type, dep_info in dependencies.items()
                if dep_info.get('from_shared', False)
            ]
            workspace_deps = [
                dep_type for dep_type, dep_info in dependencies.items()
                if dep_info.get('available', False) and not dep_info.get('from_shared', False)
            ]
            
            if shared_deps and workspace_deps:
                issues.append({
                    'type': 'mixed_dependency_sources',
                    'severity': 'info',
                    'description': 'Builder uses both workspace and shared dependencies',
                    'shared_dependencies': shared_deps,
                    'workspace_dependencies': workspace_deps
                })
            
            # Check builder class loading issues
            if not validation_results.get('builder_class_valid', False):
                issues.append({
                    'type': 'builder_class_invalid',
                    'severity': 'error',
                    'description': 'Builder class could not be loaded or is invalid'
                })
        
        except Exception as e:
            logger.warning(f"Failed to detect integration issues: {e}")
            issues.append({
                'type': 'detection_error',
                'severity': 'error',
                'description': f'Failed to detect integration issues: {e}'
            })
        
        return issues
    
    def _generate_integration_recommendations(
        self,
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on integration validation results."""
        recommendations = []
        
        try:
            issues = validation_results.get('integration_issues', [])
            
            # Recommendations for missing dependencies
            missing_dep_issues = [
                issue for issue in issues
                if issue.get('type') == 'missing_dependencies'
            ]
            
            if missing_dep_issues:
                for issue in missing_dep_issues:
                    missing_deps = issue.get('missing_dependencies', [])
                    recommendations.append(
                        f"Consider implementing workspace-specific {', '.join(missing_deps)} "
                        f"for better workspace isolation"
                    )
            
            # Recommendations for mixed usage
            mixed_usage_issues = [
                issue for issue in issues
                if issue.get('type') == 'mixed_dependency_sources'
            ]
            
            if mixed_usage_issues:
                recommendations.append(
                    "Consider using consistent dependency sources (all workspace or all shared) "
                    "for better maintainability"
                )
            
            # Recommendations for builder class issues
            builder_issues = [
                issue for issue in issues
                if issue.get('type') == 'builder_class_invalid'
            ]
            
            if builder_issues:
                recommendations.append(
                    "Verify builder class implementation and ensure it follows "
                    "the expected builder pattern"
                )
            
            # General recommendations
            if not recommendations:
                recommendations.append(
                    "Workspace builder integration looks good. "
                    "Consider adding more workspace-specific components for better isolation."
                )
        
        except Exception as e:
            logger.warning(f"Failed to generate integration recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error.")
        
        return recommendations
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about current workspace configuration."""
        return {
            'developer_id': self.developer_id,
            'workspace_root': str(self.workspace_root),
            'builder_file_path': self.builder_file_path,
            'enable_shared_fallback': self.enable_shared_fallback,
            'builder_class_name': self.builder_class.__name__ if self.builder_class else None,
            'workspace_manager_info': self.workspace_manager.get_workspace_info().model_dump(),
            'file_resolver_info': self.file_resolver.get_workspace_info(),
            'available_developers': self.workspace_manager.list_available_developers()
        }
    
    def switch_developer(self, developer_id: str, builder_file_path: Optional[str] = None) -> None:
        """Switch to a different developer workspace."""
        if developer_id == self.developer_id and builder_file_path == self.builder_file_path:
            logger.info(f"Already using developer workspace: {developer_id}")
            return
        
        logger.info(f"Switching from developer '{self.developer_id}' to '{developer_id}'")
        
        # Validate new developer exists
        available_developers = self.workspace_manager.list_available_developers()
        if developer_id not in available_developers:
            raise ValueError(f"Developer workspace not found: {developer_id}")
        
        # Update developer ID
        self.developer_id = developer_id
        
        # Update builder file path if provided
        if builder_file_path:
            self.builder_file_path = builder_file_path
        
        # Update file resolver
        self.file_resolver.switch_developer(developer_id)
        
        # Update module loader
        self.module_loader.switch_developer(developer_id)
        
        # Reload builder class from new workspace
        self.builder_class = self._load_workspace_builder_class(self.builder_file_path)
        
        # Update parent class with new builder
        self._update_builder_class(self.builder_class)
        
        logger.info(f"Successfully switched to developer workspace: {developer_id}")
    
    def _update_builder_class(self, builder_class: Type) -> None:
        """Update the parent class with new builder class."""
        # This method would update the parent class's builder_class attribute
        # The exact implementation depends on the parent class structure
        if hasattr(self, 'builder_class'):
            self.builder_class = builder_class
        
        # Re-initialize any builder-dependent attributes
        if hasattr(self, '_initialize_builder_attributes'):
            self._initialize_builder_attributes()
    
    @classmethod
    def test_all_workspace_builders(
        cls,
        workspace_root: Union[str, Path],
        developer_id: str,
        test_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover and test all builders in a workspace.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            developer_id: Specific developer workspace to target
            test_config: Test configuration parameters
            **kwargs: Additional arguments passed to individual tests
            
        Returns:
            Comprehensive test results for all workspace builders
        """
        logger.info(f"Testing all builders for developer '{developer_id}'")
        
        try:
            # Initialize workspace infrastructure
            workspace_manager = WorkspaceManager(workspace_root=workspace_root)
            file_resolver = DeveloperWorkspaceFileResolver(
                workspace_root=workspace_root,
                developer_id=developer_id
            )
            
            # Discover all builders in workspace
            builders = cls._discover_workspace_builders(workspace_manager, developer_id)
            
            if not builders:
                logger.warning(f"No builders found for developer '{developer_id}'")
                return {
                    'success': True,
                    'total_builders': 0,
                    'tested_builders': 0,
                    'results': {},
                    'workspace_metadata': {
                        'developer_id': developer_id,
                        'workspace_root': str(workspace_root)
                    }
                }
            
            # Test each builder
            all_results = {}
            successful_tests = 0
            
            for builder_name, builder_file_path in builders.items():
                logger.info(f"Testing builder: {builder_name}")
                
                try:
                    # Create test instance for this builder
                    builder_test = cls(
                        workspace_root=workspace_root,
                        developer_id=developer_id,
                        builder_file_path=builder_file_path,
                        **kwargs
                    )
                    
                    # Run test
                    test_result = builder_test.run_workspace_builder_test(
                        test_config=test_config
                    )
                    
                    all_results[builder_name] = test_result
                    
                    if test_result.get('success', False):
                        successful_tests += 1
                    
                except Exception as e:
                    logger.error(f"Failed to test builder '{builder_name}': {e}")
                    all_results[builder_name] = {
                        'success': False,
                        'error': str(e),
                        'builder_file_path': builder_file_path
                    }
            
            # Generate comprehensive report
            comprehensive_results = {
                'success': True,
                'total_builders': len(builders),
                'tested_builders': len(all_results),
                'successful_tests': successful_tests,
                'failed_tests': len(all_results) - successful_tests,
                'results': all_results,
                'workspace_metadata': {
                    'developer_id': developer_id,
                    'workspace_root': str(workspace_root),
                    'discovered_builders': list(builders.keys())
                },
                'summary': cls._generate_test_summary(all_results)
            }
            
            logger.info(f"Completed testing all builders for developer '{developer_id}': "
                       f"{successful_tests}/{len(all_results)} successful")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Failed to test all workspace builders for developer '{developer_id}': {e}")
            return {
                'success': False,
                'error': str(e),
                'workspace_metadata': {
                    'developer_id': developer_id,
                    'workspace_root': str(workspace_root)
                }
            }
    
    @classmethod
    def _discover_workspace_builders(
        cls,
        workspace_manager: WorkspaceManager,
        developer_id: str
    ) -> Dict[str, str]:
        """Discover all builders in the specified workspace."""
        builders = {}
        
        try:
            workspace_info = workspace_manager.get_workspace_info()
            if not workspace_info.developers or developer_id not in workspace_info.developers:
                logger.warning(f"Developer '{developer_id}' not found in workspace")
                return builders
            
            dev_workspace = workspace_info.developers[developer_id]
            builders_dir = dev_workspace.workspace_path / "builders"
            
            if not builders_dir.exists():
                logger.warning(f"Builders directory not found: {builders_dir}")
                return builders
            
            # Discover Python builder files in workspace
            for builder_file in builders_dir.glob("*.py"):
                if builder_file.name.startswith("__"):
                    continue
                
                builder_name = builder_file.stem
                builders[builder_name] = str(builder_file)
            
            logger.info(f"Discovered {len(builders)} builders in workspace: {list(builders.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to discover workspace builders: {e}")
        
        return builders
    
    @classmethod
    def _generate_test_summary(cls, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        summary = {
            'overall_success_rate': 0.0,
            'common_issues': [],
            'recommendations': [],
            'builder_statistics': {}
        }
        
        try:
            if not all_results:
                return summary
            
            # Calculate success rate
            successful_tests = sum(1 for result in all_results.values() if result.get('success', False))
            summary['overall_success_rate'] = successful_tests / len(all_results)
            
            # Analyze common issues
            all_issues = []
            for builder_name, result in all_results.items():
                if 'workspace_validation' in result and 'integration_issues' in result['workspace_validation']:
                    issues = result['workspace_validation']['integration_issues']
                    all_issues.extend(issues)
            
            # Count issue types
            issue_counts = {}
            for issue in all_issues:
                issue_type = issue.get('type', 'unknown')
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            # Identify common issues (appearing in >25% of builders)
            threshold = len(all_results) * 0.25
            common_issues = [
                {'type': issue_type, 'count': count, 'percentage': count / len(all_results)}
                for issue_type, count in issue_counts.items()
                if count > threshold
            ]
            summary['common_issues'] = common_issues
            
            # Generate recommendations based on common issues
            recommendations = []
            for issue in common_issues:
                if issue['type'] == 'missing_dependencies':
                    recommendations.append(
                        f"Consider implementing missing dependencies for {issue['count']} builders"
                    )
                elif issue['type'] == 'mixed_dependency_sources':
                    recommendations.append(
                        f"Consider standardizing dependency sources for {issue['count']} builders"
                    )
            
            if not recommendations:
                recommendations.append("All builders are well-integrated with their workspaces")
            
            summary['recommendations'] = recommendations
            
            # Builder statistics
            summary['builder_statistics'] = {
                'total_tested': len(all_results),
                'successful': successful_tests,
                'failed': len(all_results) - successful_tests,
                'success_rate': summary['overall_success_rate']
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate test summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    @classmethod
    def create_from_workspace_manager(
        cls,
        workspace_manager: WorkspaceManager,
        developer_id: str,
        builder_file_path: str,
        **kwargs
    ) -> 'WorkspaceUniversalStepBuilderTest':
        """Create instance from existing WorkspaceManager."""
        return cls(
            workspace_root=workspace_manager.workspace_root,
            developer_id=developer_id,
            builder_file_path=builder_file_path,
            **kwargs
        )
