"""
Validation workspace functionality layer.

This module provides workspace validation and testing components
that were consolidated from src.cursus.validation.workspace during Phase 5
structural consolidation.
"""

from .test_manager import WorkspaceTestManager
from .cross_workspace_validator import CrossWorkspaceValidator
from .test_isolation import TestWorkspaceIsolationManager
from .unified_validation_core import UnifiedValidationCore
from .workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
from .workspace_builder_test import WorkspaceUniversalStepBuilderTest
from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_manager import WorkspaceManager
from .workspace_module_loader import WorkspaceModuleLoader
from .workspace_orchestrator import WorkspaceValidationOrchestrator
from .workspace_type_detector import WorkspaceTypeDetector
from .unified_report_generator import UnifiedReportGenerator
from .unified_result_structures import ValidationSummary, WorkspaceValidationResult, UnifiedValidationResult, ValidationResultBuilder
from .legacy_adapters import LegacyWorkspaceValidationAdapter

__all__ = [
    'WorkspaceTestManager',
    'CrossWorkspaceValidator',
    'TestWorkspaceIsolationManager',
    'UnifiedValidationCore',
    'WorkspaceUnifiedAlignmentTester',
    'WorkspaceUniversalStepBuilderTest',
    'DeveloperWorkspaceFileResolver',
    'WorkspaceManager',
    'WorkspaceModuleLoader',
    'WorkspaceValidationOrchestrator',
    'WorkspaceTypeDetector',
    'UnifiedReportGenerator',
    'ValidationSummary',
    'WorkspaceValidationResult',
    'UnifiedValidationResult',
    'ValidationResultBuilder',
    'LegacyWorkspaceValidationAdapter'
]
