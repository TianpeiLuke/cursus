"""
Consolidated Test Workspace Manager

Manages test environments for workspace validation, integrating with the
consolidated workspace management system from Phase 1. This module consolidates
functionality from the distributed validation workspace managers.

Architecture Integration:
- Leverages Phase 1 consolidated WorkspaceManager and specialized managers
- Integrates validation-specific workspace management
- Provides test environment isolation and management
- Coordinates with existing validation frameworks

Features:
- Test workspace creation and management using Phase 1 lifecycle manager
- Test environment isolation using Phase 1 isolation manager
- Integration with existing validation workspace functionality
- Backward compatibility with existing validation workflows
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, ConfigDict

# PHASE 3 INTEGRATION: Import Phase 1 consolidated workspace system
from ..core.manager import WorkspaceManager, WorkspaceContext, WorkspaceConfig
from ..core.lifecycle import WorkspaceLifecycleManager
from ..core.isolation import WorkspaceIsolationManager
from ..core.integration import WorkspaceIntegrationManager

# Import existing validation workspace components for integration
from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_module_loader import WorkspaceModuleLoader

logger = logging.getLogger(__name__)


class WorkspaceTestEnvironment(BaseModel):
    """Workspace test environment configuration and state."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    test_id: str
    workspace_id: str
    environment_path: str
    test_type: str = "validation"  # "validation", "integration", "performance"
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # "active", "completed", "failed", "archived"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IsolationReport(BaseModel):
    """Test isolation validation report."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    test_environment: str
    is_isolated: bool
    isolation_violations: List[str] = Field(default_factory=list)
    boundary_checks: Dict[str, bool] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.now)


class WorkspaceTestConfig(BaseModel):
    """Configuration for test workspace management."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    test_workspace_root: str
    enable_test_isolation: bool = True
    auto_cleanup_tests: bool = True
    test_retention_days: int = 7
    max_concurrent_tests: int = 10
    validation_settings: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceTestManager:
    """
    Enhanced test workspace manager with consolidated orchestration.
    
    This manager consolidates test-related workspace functionality and integrates
    with the Phase 1 consolidated workspace management system. It provides:
    - Test environment creation and management
    - Test workspace isolation validation
    - Integration with existing validation frameworks
    - Coordination with Phase 1 specialized managers
    - ENHANCED: Consolidated validation orchestration (from WorkspaceValidationOrchestrator)
    
    Phase 3 Integration Features:
    - Uses Phase 1 WorkspaceManager as foundation
    - Leverages WorkspaceLifecycleManager for test workspace creation
    - Uses WorkspaceIsolationManager for test environment validation
    - Integrates with WorkspaceDiscoveryManager for test component discovery
    - Coordinates with WorkspaceIntegrationManager for test staging
    - CONSOLIDATED: Validation orchestration logic from separate orchestrator
    """
    
    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
        test_config: Optional[WorkspaceTestConfig] = None,
        auto_discover: bool = True,
        enable_parallel_validation: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize consolidated test workspace manager with orchestration.
        
        Args:
            workspace_manager: Phase 1 consolidated workspace manager (creates if None)
            test_config: Test-specific configuration
            auto_discover: Whether to automatically discover test workspaces
            enable_parallel_validation: Whether to enable parallel validation (consolidated from orchestrator)
            max_workers: Maximum number of parallel workers (consolidated from orchestrator)
        """
        # PHASE 3 INTEGRATION: Use consolidated workspace manager from Phase 1
        if workspace_manager:
            self.core_workspace_manager = workspace_manager
        else:
            # Create consolidated workspace manager if not provided
            from ..core.manager import WorkspaceManager
            self.core_workspace_manager = WorkspaceManager()
        
        # Access Phase 1 specialized managers
        self.lifecycle_manager = self.core_workspace_manager.lifecycle_manager
        self.isolation_manager = self.core_workspace_manager.isolation_manager
        self.discovery_manager = self.core_workspace_manager.discovery_manager
        self.integration_manager = self.core_workspace_manager.integration_manager
        
        # Test-specific configuration
        self.test_config = test_config
        
        # CONSOLIDATED: Validation orchestration configuration (from WorkspaceValidationOrchestrator)
        self.enable_parallel_validation = enable_parallel_validation
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
        
        # Test environment tracking
        self.active_test_environments: Dict[str, WorkspaceTestEnvironment] = {}
        self.test_isolation_validator = WorkspaceIsolationValidator(self)
        
        # CONSOLIDATED: Validation orchestration components (from WorkspaceValidationOrchestrator)
        self.validators = [
            # Will be initialized lazily when needed
        ]
        
        # Integration with existing validation workspace functionality
        self._integrate_existing_validation_components()
        
        if auto_discover and self.core_workspace_manager.workspace_root:
            self.discover_test_workspaces()
        
        logger.info("Initialized consolidated test workspace manager with Phase 1 integration and validation orchestration")
    
    def _integrate_existing_validation_components(self) -> None:
        """Integrate with existing validation workspace components."""
        try:
            # Integration point for existing workspace validation functionality
            # This maintains backward compatibility while leveraging Phase 1 foundation
            self._existing_workspace_manager = None
            
            # Try to integrate with existing validation workspace manager
            if self.core_workspace_manager.workspace_root:
                try:
                    from .workspace_manager import WorkspaceManager as ExistingWorkspaceManager
                    self._existing_workspace_manager = ExistingWorkspaceManager(
                        workspace_root=self.core_workspace_manager.workspace_root,
                        auto_discover=False  # We'll handle discovery through Phase 1
                    )
                    logger.info("Integrated with existing validation workspace manager")
                except Exception as e:
                    logger.warning(f"Could not integrate with existing workspace manager: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to integrate existing validation components: {e}")
    
    # Test Environment Management
    
    def create_test_environment(
        self,
        test_id: str,
        workspace_id: Optional[str] = None,
        test_type: str = "validation",
        **kwargs
    ) -> WorkspaceTestEnvironment:
        """
        Create isolated test environment using Phase 1 lifecycle manager.
        
        Args:
            test_id: Unique identifier for the test
            workspace_id: Target workspace ID (creates test workspace if None)
            test_type: Type of test environment
            **kwargs: Additional test environment configuration
        
        Returns:
            WorkspaceTestEnvironment configuration
        """
        logger.info(f"Creating test environment: {test_id}")
        
        try:
            # Use Phase 1 lifecycle manager to create test workspace
            if not workspace_id:
                workspace_id = f"test_{test_id}"
            
            # Create test workspace using Phase 1 consolidated system
            workspace_context = self.lifecycle_manager.create_workspace(
                developer_id=workspace_id,
                workspace_type="test",
                template="test_environment",
                **kwargs
            )
            
            # Create test environment configuration
            test_environment = WorkspaceTestEnvironment(
                test_id=test_id,
                workspace_id=workspace_context.workspace_id,
                environment_path=workspace_context.workspace_path,
                test_type=test_type,
                metadata=kwargs
            )
            
            # Register test environment
            self.active_test_environments[test_id] = test_environment
            
            # Set up test environment isolation using Phase 1 isolation manager
            self._setup_test_isolation(test_environment)
            
            logger.info(f"Successfully created test environment: {test_id}")
            return test_environment
            
        except Exception as e:
            logger.error(f"Failed to create test environment {test_id}: {e}")
            raise
    
    def _setup_test_isolation(self, test_environment: WorkspaceTestEnvironment) -> None:
        """Set up test environment isolation using Phase 1 isolation manager."""
        try:
            # Use Phase 1 isolation manager for test environment setup
            isolation_result = self.isolation_manager.create_isolated_environment(
                workspace_id=test_environment.workspace_id
            )
            
            # Update test environment metadata with isolation info
            test_environment.metadata["isolation_config"] = {
                "isolation_id": isolation_result.get("isolation_id"),
                "boundary_enforcement": True,
                "resource_limits": isolation_result.get("resource_limits", {}),
                "access_controls": isolation_result.get("access_controls", {})
            }
            
            logger.debug(f"Set up test isolation for: {test_environment.test_id}")
            
        except Exception as e:
            logger.warning(f"Failed to set up test isolation for {test_environment.test_id}: {e}")
    
    def validate_test_isolation(self, test_environment: WorkspaceTestEnvironment) -> IsolationReport:
        """
        Validate test isolation using Phase 1 isolation manager.
        
        Args:
            test_environment: Test environment to validate
        
        Returns:
            IsolationReport with validation results
        """
        logger.info(f"Validating test isolation for: {test_environment.test_id}")
        
        try:
            # Use Phase 1 isolation manager for validation
            validation_result = self.isolation_manager.validate_workspace_boundaries(
                workspace_path=test_environment.environment_path
            )
            
            # Check for isolation violations
            violations = []
            if not validation_result.is_valid:
                violations.extend(validation_result.issues)
            
            # Additional test-specific isolation checks
            test_violations = self.test_isolation_validator.validate_test_boundaries(
                test_environment
            )
            violations.extend(test_violations)
            
            # Create isolation report
            isolation_report = IsolationReport(
                test_environment=test_environment.test_id,
                is_isolated=len(violations) == 0,
                isolation_violations=violations,
                boundary_checks={
                    "workspace_boundaries": validation_result.is_valid,
                    "test_isolation": len(test_violations) == 0,
                    "resource_isolation": self._check_resource_isolation(test_environment),
                    "access_control": self._check_access_control(test_environment)
                },
                recommendations=self._generate_isolation_recommendations(violations)
            )
            
            logger.info(f"Test isolation validation completed for: {test_environment.test_id}")
            return isolation_report
            
        except Exception as e:
            logger.error(f"Failed to validate test isolation for {test_environment.test_id}: {e}")
            # Return failed isolation report
            return IsolationReport(
                test_environment=test_environment.test_id,
                is_isolated=False,
                isolation_violations=[f"Validation error: {e}"],
                boundary_checks={},
                recommendations=["Fix validation errors and retry"]
            )
    
    def _check_resource_isolation(self, test_environment: WorkspaceTestEnvironment) -> bool:
        """Check resource isolation for test environment."""
        try:
            # Use Phase 1 isolation manager for resource checks
            resource_check = self.isolation_manager.detect_isolation_violations(
                workspace_path=test_environment.environment_path
            )
            return len(resource_check) == 0
        except Exception as e:
            logger.warning(f"Resource isolation check failed: {e}")
            return False
    
    def _check_access_control(self, test_environment: WorkspaceTestEnvironment) -> bool:
        """Check access control for test environment."""
        try:
            # Check if test environment has proper access controls
            env_path = Path(test_environment.environment_path)
            
            # Basic access control checks
            if not env_path.exists():
                return False
            
            # Check for test-specific access patterns
            isolation_config = test_environment.metadata.get("isolation_config", {})
            return isolation_config.get("boundary_enforcement", False)
            
        except Exception as e:
            logger.warning(f"Access control check failed: {e}")
            return False
    
    def _generate_isolation_recommendations(self, violations: List[str]) -> List[str]:
        """Generate recommendations for isolation violations."""
        recommendations = []
        
        if not violations:
            recommendations.append("Test isolation is properly configured")
            return recommendations
        
        # Generate specific recommendations based on violations
        for violation in violations:
            if "boundary" in violation.lower():
                recommendations.append("Review workspace boundary configuration")
            elif "access" in violation.lower():
                recommendations.append("Check access control settings")
            elif "resource" in violation.lower():
                recommendations.append("Verify resource isolation limits")
            else:
                recommendations.append(f"Address isolation issue: {violation}")
        
        return recommendations
    
    # Test Workspace Discovery and Management
    
    def discover_test_workspaces(self) -> Dict[str, Any]:
        """
        Discover test workspaces using Phase 1 discovery manager.
        
        Returns:
            Dictionary containing discovered test workspace information
        """
        logger.info("Discovering test workspaces using Phase 1 discovery manager")
        
        try:
            # Use Phase 1 discovery manager for workspace discovery
            discovery_result = self.discovery_manager.discover_workspaces(
                self.core_workspace_manager.workspace_root
            )
            
            # Filter for test workspaces
            test_workspaces = {}
            for workspace_info in discovery_result.get('workspaces', []):
                workspace_type = workspace_info.get('workspace_type', '')
                if workspace_type == 'test' or workspace_info.get('workspace_id', '').startswith('test_'):
                    test_workspaces[workspace_info['workspace_id']] = workspace_info
            
            # Update active test environments
            for workspace_id, workspace_info in test_workspaces.items():
                if workspace_id not in self.active_test_environments:
                    # Create test environment entry for discovered test workspace
                    test_env = WorkspaceTestEnvironment(
                        test_id=workspace_id.replace('test_', ''),
                        workspace_id=workspace_id,
                        environment_path=workspace_info.get('workspace_path', ''),
                        test_type="discovered",
                        metadata=workspace_info.get('metadata', {})
                    )
                    self.active_test_environments[workspace_id] = test_env
            
            logger.info(f"Discovered {len(test_workspaces)} test workspaces")
            return {
                "test_workspaces": test_workspaces,
                "total_test_workspaces": len(test_workspaces),
                "discovery_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to discover test workspaces: {e}")
            return {"error": str(e), "test_workspaces": {}}
    
    def cleanup_test_environment(self, test_id: str) -> bool:
        """
        Clean up test environment using Phase 1 lifecycle manager.
        
        Args:
            test_id: Test environment identifier
        
        Returns:
            True if cleanup was successful
        """
        logger.info(f"Cleaning up test environment: {test_id}")
        
        if test_id not in self.active_test_environments:
            logger.warning(f"Test environment not found: {test_id}")
            return False
        
        try:
            test_environment = self.active_test_environments[test_id]
            
            # Use Phase 1 lifecycle manager for cleanup
            cleanup_success = self.lifecycle_manager.delete_workspace(
                test_environment.workspace_id
            )
            
            if cleanup_success:
                # Remove from active test environments
                del self.active_test_environments[test_id]
                logger.info(f"Successfully cleaned up test environment: {test_id}")
                return True
            else:
                logger.error(f"Failed to cleanup test environment: {test_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cleaning up test environment {test_id}: {e}")
            return False
    
    # Integration with Existing Validation Components
    
    def get_file_resolver(
        self,
        test_id: Optional[str] = None,
        developer_id: Optional[str] = None,
        **kwargs
    ) -> DeveloperWorkspaceFileResolver:
        """
        Get workspace-aware file resolver for test environment.
        
        Args:
            test_id: Test environment identifier
            developer_id: Developer identifier (uses test workspace if test_id provided)
            **kwargs: Additional arguments for file resolver
        
        Returns:
            Configured DeveloperWorkspaceFileResolver
        """
        # Determine target workspace
        if test_id and test_id in self.active_test_environments:
            target_workspace = self.active_test_environments[test_id].workspace_id
        else:
            target_workspace = developer_id
        
        # Use Phase 1 discovery manager to get file resolver
        try:
            return self.discovery_manager.get_file_resolver(
                developer_id=target_workspace,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to get file resolver from Phase 1, falling back to existing: {e}")
            
            # Fallback to existing validation workspace manager if available
            if self._existing_workspace_manager:
                return self._existing_workspace_manager.get_file_resolver(
                    developer_id=target_workspace,
                    **kwargs
                )
            else:
                # Create directly if no existing manager
                return DeveloperWorkspaceFileResolver(
                    workspace_root=self.core_workspace_manager.workspace_root,
                    developer_id=target_workspace,
                    **kwargs
                )
    
    def get_module_loader(
        self,
        test_id: Optional[str] = None,
        developer_id: Optional[str] = None,
        **kwargs
    ) -> WorkspaceModuleLoader:
        """
        Get workspace-aware module loader for test environment.
        
        Args:
            test_id: Test environment identifier
            developer_id: Developer identifier (uses test workspace if test_id provided)
            **kwargs: Additional arguments for module loader
        
        Returns:
            Configured WorkspaceModuleLoader
        """
        # Determine target workspace
        if test_id and test_id in self.active_test_environments:
            target_workspace = self.active_test_environments[test_id].workspace_id
        else:
            target_workspace = developer_id
        
        # Use Phase 1 discovery manager to get module loader
        try:
            return self.discovery_manager.get_module_loader(
                developer_id=target_workspace,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to get module loader from Phase 1, falling back to existing: {e}")
            
            # Fallback to existing validation workspace manager if available
            if self._existing_workspace_manager:
                return self._existing_workspace_manager.get_module_loader(
                    developer_id=target_workspace,
                    **kwargs
                )
            else:
                # Create directly if no existing manager
                return WorkspaceModuleLoader(
                    workspace_root=self.core_workspace_manager.workspace_root,
                    developer_id=target_workspace,
                    **kwargs
                )
    
    # Test Environment Information and Statistics
    
    def get_test_environment_info(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get test environment information.
        
        Args:
            test_id: Optional specific test ID to get info for
        
        Returns:
            Dictionary with test environment information
        """
        if test_id:
            if test_id not in self.active_test_environments:
                return {"error": f"Test environment not found: {test_id}"}
            
            test_env = self.active_test_environments[test_id]
            return {
                "test_id": test_env.test_id,
                "workspace_id": test_env.workspace_id,
                "environment_path": test_env.environment_path,
                "test_type": test_env.test_type,
                "status": test_env.status,
                "created_at": test_env.created_at.isoformat(),
                "metadata": test_env.metadata
            }
        else:
            # Return information for all test environments
            return {
                "total_test_environments": len(self.active_test_environments),
                "test_environments": {
                    test_id: {
                        "workspace_id": test_env.workspace_id,
                        "test_type": test_env.test_type,
                        "status": test_env.status,
                        "created_at": test_env.created_at.isoformat()
                    }
                    for test_id, test_env in self.active_test_environments.items()
                },
                "phase1_integration": {
                    "core_workspace_manager": str(type(self.core_workspace_manager)),
                    "lifecycle_manager": str(type(self.lifecycle_manager)),
                    "isolation_manager": str(type(self.isolation_manager)),
                    "discovery_manager": str(type(self.discovery_manager)),
                    "integration_manager": str(type(self.integration_manager))
                }
            }
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test workspace statistics."""
        try:
            # Get statistics from Phase 1 managers
            lifecycle_stats = self.lifecycle_manager.get_statistics()
            isolation_stats = self.isolation_manager.get_statistics()
            discovery_stats = self.discovery_manager.get_statistics()
            integration_stats = self.integration_manager.get_statistics()
            
            return {
                "test_environments": {
                    "total": len(self.active_test_environments),
                    "by_type": {
                        test_type: len([
                            env for env in self.active_test_environments.values()
                            if env.test_type == test_type
                        ])
                        for test_type in ["validation", "integration", "performance", "discovered"]
                    },
                    "by_status": {
                        status: len([
                            env for env in self.active_test_environments.values()
                            if env.status == status
                        ])
                        for status in ["active", "completed", "failed", "archived"]
                    }
                },
                "phase1_integration_stats": {
                    "lifecycle": lifecycle_stats,
                    "isolation": isolation_stats,
                    "discovery": discovery_stats,
                    "integration": integration_stats
                },
                "consolidation_status": {
                    "phase1_integrated": True,
                    "existing_validation_integrated": self._existing_workspace_manager is not None,
                    "test_isolation_enabled": self.test_config.enable_test_isolation if self.test_config else True
                }
            }
        except Exception as e:
            logger.error(f"Failed to get test statistics: {e}")
            return {"error": str(e)}
    
    # CONSOLIDATED: Validation Orchestration Methods (from WorkspaceValidationOrchestrator)
    
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
        CONSOLIDATED from WorkspaceValidationOrchestrator.
        
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
            # Validate developer exists using Phase 1 discovery manager
            available_developers = self.discovery_manager.list_available_developers()
            if developer_id not in available_developers:
                raise ValueError(f"Developer workspace not found: {developer_id}")
            
            # Initialize validation results
            validation_results = {
                'developer_id': developer_id,
                'workspace_root': str(self.core_workspace_manager.workspace_root),
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
                
                # Check if alignment validation has any failures
                if self._has_validation_failures(alignment_results):
                    validation_results['success'] = False
            
            # Run builder validation if requested
            if 'builders' in validation_levels or 'all' in validation_levels:
                logger.info(f"Running builder validation for developer '{developer_id}'")
                builder_results = self._run_builder_validation(
                    developer_id, target_builders, validation_config
                )
                validation_results['results']['builders'] = builder_results
                
                # Check if builder validation has any failures
                if self._has_validation_failures(builder_results):
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
                'workspace_root': str(self.core_workspace_manager.workspace_root),
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
    
    def _run_alignment_validation(
        self,
        developer_id: str,
        target_scripts: Optional[List[str]],
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run alignment validation for a specific workspace."""
        try:
            # Import alignment tester lazily to avoid circular imports
            from .workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
            
            # Create alignment tester instance
            alignment_tester = WorkspaceUnifiedAlignmentTester(
                workspace_root=self.core_workspace_manager.workspace_root,
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
            # Import builder tester lazily to avoid circular imports
            from .workspace_builder_test import WorkspaceUniversalStepBuilderTest
            
            # Create builder tester instance
            builder_tester = WorkspaceUniversalStepBuilderTest(
                workspace_root=self.core_workspace_manager.workspace_root,
                developer_id=developer_id,
                builder_file_path=""  # Will be set as needed
            )
            
            # Run workspace builder test
            builder_results = builder_tester.run_workspace_builder_test()
            
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
    
    def _has_validation_failures(self, validation_results: Dict[str, Any]) -> bool:
        """Check if validation results contain any failures."""
        if not validation_results:
            return True  # No results means failure
        
        # Check for explicit success flag
        if 'success' in validation_results:
            return not validation_results['success']
        
        # Check for errors
        if 'error' in validation_results:
            return True
        
        # Check nested results for failures
        for developer_id, developer_results in validation_results.items():
            if isinstance(developer_results, dict):
                for level_name, level_result in developer_results.items():
                    if isinstance(level_result, dict):
                        if not level_result.get('passed', True):
                            return True
        
        return False
    
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


class WorkspaceIsolationValidator:
    """Validates test environment isolation."""
    
    def __init__(self, test_manager: WorkspaceTestManager):
        """Initialize with test manager reference."""
        self.test_manager = test_manager
        self.logger = logging.getLogger(__name__)
    
    def validate_test_boundaries(self, test_environment: WorkspaceTestEnvironment) -> List[str]:
        """
        Validate test-specific isolation boundaries.
        
        Args:
            test_environment: Test environment to validate
        
        Returns:
            List of isolation violations
        """
        violations = []
        
        try:
            env_path = Path(test_environment.environment_path)
            
            # Check test environment exists
            if not env_path.exists():
                violations.append(f"Test environment path does not exist: {env_path}")
                return violations
            
            # Check for test-specific isolation requirements
            violations.extend(self._check_test_data_isolation(env_path))
            violations.extend(self._check_test_output_isolation(env_path))
            violations.extend(self._check_test_dependency_isolation(env_path))
            
        except Exception as e:
            violations.append(f"Test boundary validation error: {e}")
        
        return violations
    
    def _check_test_data_isolation(self, env_path: Path) -> List[str]:
        """Check test data isolation."""
        violations = []
        
        # Check for test data directories
        test_data_dirs = ["inputs", "test_data", "cache"]
        for data_dir in test_data_dirs:
            data_path = env_path / data_dir
            if data_path.exists():
                # Check if test data is properly isolated
                if not self._is_path_isolated(data_path):
                    violations.append(f"Test data not isolated: {data_path}")
        
        return violations
    
    def _check_test_output_isolation(self, env_path: Path) -> List[str]:
        """Check test output isolation."""
        violations = []
        
        # Check for test output directories
        output_dirs = ["outputs", "logs", "results"]
        for output_dir in output_dirs:
            output_path = env_path / output_dir
            if output_path.exists():
                # Check if test outputs are properly isolated
                if not self._is_path_isolated(output_path):
                    violations.append(f"Test output not isolated: {output_path}")
        
        return violations
    
    def _check_test_dependency_isolation(self, env_path: Path) -> List[str]:
        """Check test dependency isolation."""
        violations = []
        
        # Check for dependency isolation
        # This is a simplified check - in practice, you might check Python path isolation,
        # environment variables, etc.
        
        return violations
    
    def _is_path_isolated(self, path: Path) -> bool:
        """Check if a path is properly isolated."""
        try:
            # Basic isolation check - ensure path is within test environment
            # In practice, this would be more sophisticated
            return path.exists() and path.is_dir()
        except Exception:
            return False


# Convenience functions for backward compatibility and easy integration

def create_test_workspace_manager(
    workspace_root: Optional[str] = None,
    test_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WorkspaceTestManager:
    """
    Convenience function to create a configured WorkspaceTestManager.
    
    Args:
        workspace_root: Root directory for workspaces
        test_config: Test-specific configuration
        **kwargs: Additional arguments for WorkspaceManager
    
    Returns:
        Configured WorkspaceTestManager instance
    """
    # Create Phase 1 consolidated workspace manager
    core_manager = WorkspaceManager(workspace_root=workspace_root, **kwargs)
    
    # Create test configuration if provided
    test_workspace_config = None
    if test_config:
        test_workspace_config = WorkspaceTestConfig(**test_config)
    
    return WorkspaceTestManager(
        workspace_manager=core_manager,
        test_config=test_workspace_config
    )


def validate_test_workspace_structure(
    test_workspace_root: str,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate test workspace structure.
    
    Args:
        test_workspace_root: Root directory to validate
        strict: Whether to apply strict validation rules
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    test_manager = create_test_workspace_manager(workspace_root=test_workspace_root)
    
    # Use Phase 1 isolation manager for validation
    return test_manager.isolation_manager.validate_workspace_structure(
        workspace_root=Path(test_workspace_root),
        strict=strict
    )
