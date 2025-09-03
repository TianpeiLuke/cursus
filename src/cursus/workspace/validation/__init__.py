"""
Validation workspace functionality layer.

This module provides workspace validation and testing components
that were consolidated from src.cursus.validation.workspace during Phase 5
structural consolidation.
"""

from .workspace_test_manager import WorkspaceTestManager
from .cross_workspace_validator import CrossWorkspaceValidator
from .workspace_isolation import WorkspaceTestIsolationManager
from .unified_validation_core import UnifiedValidationCore
from .workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
from .workspace_builder_test import WorkspaceUniversalStepBuilderTest
from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_manager import WorkspaceManager
from .workspace_module_loader import WorkspaceModuleLoader
# PHASE 1 CONSOLIDATION: WorkspaceValidationOrchestrator removed, functionality moved to WorkspaceTestManager
from .workspace_type_detector import WorkspaceTypeDetector
from .unified_report_generator import UnifiedReportGenerator
from .unified_result_structures import (
    BaseValidationResult,
    WorkspaceValidationResult, 
    AlignmentTestResult,
    BuilderTestResult,
    IsolationTestResult,
    ValidationSummary,
    UnifiedValidationResult,
    ValidationResultBuilder,
    create_single_workspace_result,
    create_empty_result
)
from .legacy_adapters import LegacyWorkspaceValidationAdapter

__all__ = [
    # PHASE 1 CONSOLIDATED: Enhanced WorkspaceTestManager with orchestration
    'WorkspaceTestManager',
    'CrossWorkspaceValidator',
    'WorkspaceTestIsolationManager',
    'UnifiedValidationCore',
    'WorkspaceUnifiedAlignmentTester',
    'WorkspaceUniversalStepBuilderTest',
    'DeveloperWorkspaceFileResolver',
    'WorkspaceManager',
    'WorkspaceModuleLoader',
    # NOTE: WorkspaceValidationOrchestrator functionality consolidated into WorkspaceTestManager
    'WorkspaceTypeDetector',
    'UnifiedReportGenerator',
    # PHASE 1 CONSOLIDATED: Enhanced result structures with inheritance
    'BaseValidationResult',
    'WorkspaceValidationResult',
    'AlignmentTestResult',
    'BuilderTestResult',
    'IsolationTestResult',
    'ValidationSummary',
    'UnifiedValidationResult',
    'ValidationResultBuilder',
    'create_single_workspace_result',
    'create_empty_result',
    'LegacyWorkspaceValidationAdapter'
]
