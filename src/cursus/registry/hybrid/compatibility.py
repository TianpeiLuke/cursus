"""
Hybrid Registry Backward Compatibility Layer

This module provides backward compatibility for the hybrid registry system,
ensuring existing code continues to work while providing migration paths
to the new hybrid architecture.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Callable
from functools import wraps
import inspect

from .models import StepDefinition, NamespacedStepDefinition, ResolutionContext
from .manager import HybridRegistryManager, RegistryConfig
from .utils import StepDefinitionConverter, RegistryErrorFormatter

logger = logging.getLogger(__name__)


class LegacyRegistryAdapter:
    """
    Adapter that provides the legacy registry interface while using the hybrid system.
    
    This adapter ensures that existing code using the old registry API continues
    to work without modification while leveraging the new hybrid capabilities.
    """
    
    def __init__(self, hybrid_manager: HybridRegistryManager):
        self.hybrid_manager = hybrid_manager
        self._converter = StepDefinitionConverter()
        self._error_formatter = RegistryErrorFormatter()
        self._default_context = ResolutionContext(
            workspace_id="legacy",
            resolution_strategy="core_fallback"
        )
    
    def get_step_builder(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Legacy method to get step builder in old format."""
        warnings.warn(
            "get_step_builder is deprecated. Use hybrid_manager.get_step() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        result = self.hybrid_manager.get_step(step_name, self._default_context)
        if result.step_definition:
            return self._converter.to_legacy_format(result.step_definition)
        return None
    
    def list_step_builders(self) -> List[str]:
        """Legacy method to list all step builders."""
        warnings.warn(
            "list_step_builders is deprecated. Use hybrid_manager.list_all_steps() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        return self.hybrid_manager.list_all_steps()
    
    def has_step_builder(self, step_name: str) -> bool:
        """Legacy method to check if step builder exists."""
        warnings.warn(
            "has_step_builder is deprecated. Use hybrid_manager.get_step() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        result = self.hybrid_manager.get_step(step_name, self._default_context)
        return result.step_definition is not None
    
    def get_registry_dict(self) -> Dict[str, Dict[str, Any]]:
        """Legacy method to get entire registry as dictionary."""
        warnings.warn(
            "get_registry_dict is deprecated. Use hybrid_manager methods instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        registry_dict = {}
        step_names = self.hybrid_manager.list_all_steps()
        
        for step_name in step_names:
            result = self.hybrid_manager.get_step(step_name, self._default_context)
            if result.step_definition:
                registry_dict[step_name] = self._converter.to_legacy_format(result.step_definition)
        
        return registry_dict


class EnhancedBackwardCompatibilityLayer:
    """
    Enhanced backward compatibility layer with intelligent API preservation.
    
    This layer provides comprehensive backward compatibility while offering
    migration assistance and performance optimizations.
    """
    
    def __init__(self, hybrid_manager: HybridRegistryManager, 
                 enable_migration_warnings: bool = True,
                 enable_performance_tracking: bool = False):
        self.hybrid_manager = hybrid_manager
        self.enable_migration_warnings = enable_migration_warnings
        self.enable_performance_tracking = enable_performance_tracking
        
        self._legacy_adapter = LegacyRegistryAdapter(hybrid_manager)
        self._converter = StepDefinitionConverter()
        self._error_formatter = RegistryErrorFormatter()
        
        # Performance tracking
        self._api_call_counts: Dict[str, int] = {}
        self._migration_suggestions: Dict[str, str] = {}
        
        # Legacy API mappings
        self._legacy_methods = {
            'get_step_builder': self._legacy_get_step_builder,
            'list_step_builders': self._legacy_list_step_builders,
            'has_step_builder': self._legacy_has_step_builder,
            'get_registry_dict': self._legacy_get_registry_dict,
            'load_registry': self._legacy_load_registry,
            'reload_registry': self._legacy_reload_registry
        }
    
    def __getattr__(self, name: str) -> Any:
        """Dynamic method resolution for legacy API calls."""
        if name in self._legacy_methods:
            if self.enable_performance_tracking:
                self._track_api_call(name)
            
            if self.enable_migration_warnings:
                self._suggest_migration(name)
            
            return self._legacy_methods[name]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _track_api_call(self, method_name: str) -> None:
        """Track legacy API usage for migration planning."""
        self._api_call_counts[method_name] = self._api_call_counts.get(method_name, 0) + 1
    
    def _suggest_migration(self, method_name: str) -> None:
        """Provide migration suggestions for legacy API calls."""
        if method_name not in self._migration_suggestions:
            migration_map = {
                'get_step_builder': 'Use hybrid_manager.get_step() for better conflict resolution',
                'list_step_builders': 'Use hybrid_manager.list_all_steps() for workspace-aware listing',
                'has_step_builder': 'Use hybrid_manager.get_step() and check result.step_definition',
                'get_registry_dict': 'Use hybrid_manager methods for better performance',
                'load_registry': 'Use hybrid_manager.load_all_registries() for comprehensive loading',
                'reload_registry': 'Use hybrid_manager.core_registry.reload_registry() for targeted reloading'
            }
            
            suggestion = migration_map.get(method_name, "Consider migrating to hybrid registry API")
            self._migration_suggestions[method_name] = suggestion
            
            logger.info(f"Migration suggestion for {method_name}: {suggestion}")
    
    def _legacy_get_step_builder(self, step_name: str, workspace_id: str = "legacy") -> Optional[Dict[str, Any]]:
        """Enhanced legacy get_step_builder with workspace support."""
        context = ResolutionContext(
            workspace_id=workspace_id,
            resolution_strategy="core_fallback"
        )
        
        result = self.hybrid_manager.get_step(step_name, context)
        if result.step_definition:
            legacy_format = self._converter.to_legacy_format(result.step_definition)
            
            # Add hybrid metadata for debugging
            legacy_format['_hybrid_metadata'] = {
                'source_registry': result.source_registry,
                'workspace_id': result.workspace_id,
                'conflict_detected': result.conflict_detected,
                'resolution_strategy': result.resolution_strategy
            }
            
            return legacy_format
        return None
    
    def _legacy_list_step_builders(self, workspace_id: str = "legacy") -> List[str]:
        """Enhanced legacy list_step_builders with workspace filtering."""
        if workspace_id == "legacy":
            return self.hybrid_manager.list_all_steps()
        else:
            # Filter by workspace if specified
            all_steps_by_source = self.hybrid_manager.list_all_steps(include_source=True)
            workspace_steps = set()
            
            # Add core steps
            workspace_steps.update(all_steps_by_source.get("core", []))
            
            # Add workspace-specific steps
            workspace_steps.update(all_steps_by_source.get(workspace_id, []))
            
            return sorted(list(workspace_steps))
    
    def _legacy_has_step_builder(self, step_name: str, workspace_id: str = "legacy") -> bool:
        """Enhanced legacy has_step_builder with workspace support."""
        context = ResolutionContext(
            workspace_id=workspace_id,
            resolution_strategy="core_fallback"
        )
        
        result = self.hybrid_manager.get_step(step_name, context)
        return result.step_definition is not None
    
    def _legacy_get_registry_dict(self, workspace_id: str = "legacy") -> Dict[str, Dict[str, Any]]:
        """Enhanced legacy get_registry_dict with workspace filtering."""
        registry_dict = {}
        step_names = self._legacy_list_step_builders(workspace_id)
        
        context = ResolutionContext(
            workspace_id=workspace_id,
            resolution_strategy="core_fallback"
        )
        
        for step_name in step_names:
            result = self.hybrid_manager.get_step(step_name, context)
            if result.step_definition:
                legacy_format = self._converter.to_legacy_format(result.step_definition)
                
                # Add hybrid metadata
                legacy_format['_hybrid_metadata'] = {
                    'source_registry': result.source_registry,
                    'workspace_id': result.workspace_id,
                    'conflict_detected': result.conflict_detected
                }
                
                registry_dict[step_name] = legacy_format
        
        return registry_dict
    
    def _legacy_load_registry(self) -> bool:
        """Enhanced legacy load_registry using hybrid system."""
        results = self.hybrid_manager.load_all_registries()
        
        # Check if any registry failed to load
        for registry_name, result in results.items():
            if not result.is_valid:
                logger.error(f"Failed to load {registry_name} registry: {result.errors}")
                return False
        
        return True
    
    def _legacy_reload_registry(self) -> bool:
        """Enhanced legacy reload_registry using hybrid system."""
        try:
            # Reload core registry
            core_result = self.hybrid_manager.core_registry.reload_registry()
            if not core_result.is_valid:
                logger.error(f"Failed to reload core registry: {core_result.errors}")
                return False
            
            # Reload local registries
            for workspace_id, registry in self.hybrid_manager.local_registries.items():
                local_result = registry.reload_registry()
                if not local_result.is_valid:
                    logger.error(f"Failed to reload {workspace_id} registry: {local_result.errors}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload registries: {str(e)}")
            return False
    
    def get_migration_report(self) -> Dict[str, Any]:
        """Generate a migration report showing legacy API usage."""
        return {
            "api_call_counts": dict(self._api_call_counts),
            "migration_suggestions": dict(self._migration_suggestions),
            "total_legacy_calls": sum(self._api_call_counts.values()),
            "unique_legacy_methods": len(self._api_call_counts)
        }
    
    def reset_tracking(self) -> None:
        """Reset performance tracking data."""
        self._api_call_counts.clear()
        self._migration_suggestions.clear()


def deprecated_registry_method(replacement: str):
    """Decorator to mark registry methods as deprecated."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. Use {replacement} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class LegacyRegistryInterface:
    """
    Legacy registry interface that mimics the old API exactly.
    
    This interface provides 100% backward compatibility for existing code
    while internally using the hybrid registry system.
    """
    
    def __init__(self, registry_path: str, workspace_id: str = "legacy"):
        # Initialize hybrid system
        config = RegistryConfig(
            core_registry_path=registry_path,
            conflict_resolution_strategy="core_fallback"
        )
        self.hybrid_manager = HybridRegistryManager(config)
        self.compatibility_layer = EnhancedBackwardCompatibilityLayer(
            self.hybrid_manager,
            enable_migration_warnings=True,
            enable_performance_tracking=True
        )
        self.workspace_id = workspace_id
        
        # Load registries
        self.hybrid_manager.load_all_registries()
    
    @deprecated_registry_method("hybrid_manager.get_step()")
    def get_step_builder(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get step builder in legacy format."""
        return self.compatibility_layer._legacy_get_step_builder(step_name, self.workspace_id)
    
    @deprecated_registry_method("hybrid_manager.list_all_steps()")
    def list_step_builders(self) -> List[str]:
        """List all step builders."""
        return self.compatibility_layer._legacy_list_step_builders(self.workspace_id)
    
    @deprecated_registry_method("hybrid_manager.get_step()")
    def has_step_builder(self, step_name: str) -> bool:
        """Check if step builder exists."""
        return self.compatibility_layer._legacy_has_step_builder(step_name, self.workspace_id)
    
    @deprecated_registry_method("hybrid_manager methods")
    def get_registry_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get entire registry as dictionary."""
        return self.compatibility_layer._legacy_get_registry_dict(self.workspace_id)
    
    @deprecated_registry_method("hybrid_manager.load_all_registries()")
    def load_registry(self) -> bool:
        """Load registry."""
        return self.compatibility_layer._legacy_load_registry()
    
    @deprecated_registry_method("hybrid_manager.core_registry.reload_registry()")
    def reload_registry(self) -> bool:
        """Reload registry."""
        return self.compatibility_layer._legacy_reload_registry()
    
    def get_migration_info(self) -> Dict[str, Any]:
        """Get information about migration to hybrid system."""
        return {
            "migration_report": self.compatibility_layer.get_migration_report(),
            "hybrid_status": self.hybrid_manager.get_registry_status(),
            "recommendations": [
                "Consider migrating to HybridRegistryManager for better performance",
                "Use ResolutionContext for workspace-aware step resolution",
                "Leverage conflict resolution strategies for multi-developer environments"
            ]
        }


class APICompatibilityChecker:
    """
    Checks API compatibility between legacy and hybrid systems.
    
    This checker helps identify potential issues when migrating from
    legacy registry to hybrid registry.
    """
    
    def __init__(self):
        self._converter = StepDefinitionConverter()
        self._error_formatter = RegistryErrorFormatter()
    
    def check_step_compatibility(self, legacy_step: Dict[str, Any], 
                               hybrid_step: StepDefinition) -> Dict[str, Any]:
        """Check compatibility between legacy and hybrid step definitions."""
        
        compatibility_issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['step_type', 'script_path']
        for field in required_fields:
            if field not in legacy_step:
                compatibility_issues.append(f"Missing required field: {field}")
            elif getattr(hybrid_step, field, None) != legacy_step.get(field):
                compatibility_issues.append(f"Field mismatch for {field}")
        
        # Check hyperparameters
        legacy_hyperparams = legacy_step.get('hyperparameters', {})
        hybrid_hyperparams = hybrid_step.hyperparameters or {}
        
        if legacy_hyperparams != hybrid_hyperparams:
            warnings.append("Hyperparameter differences detected")
        
        # Check dependencies
        legacy_deps = legacy_step.get('dependencies', [])
        hybrid_deps = hybrid_step.dependencies or []
        
        if set(legacy_deps) != set(hybrid_deps):
            warnings.append("Dependency differences detected")
        
        return {
            "compatible": len(compatibility_issues) == 0,
            "issues": compatibility_issues,
            "warnings": warnings,
            "legacy_fields": list(legacy_step.keys()),
            "hybrid_fields": [f for f in dir(hybrid_step) if not f.startswith('_')]
        }
    
    def check_registry_compatibility(self, legacy_registry: Dict[str, Dict[str, Any]],
                                   hybrid_manager: HybridRegistryManager) -> Dict[str, Any]:
        """Check compatibility between entire legacy and hybrid registries."""
        
        compatibility_report = {
            "total_steps": len(legacy_registry),
            "compatible_steps": 0,
            "incompatible_steps": 0,
            "step_issues": {},
            "overall_warnings": [],
            "migration_blockers": []
        }
        
        context = ResolutionContext(workspace_id="compatibility_check")
        
        for step_name, legacy_step in legacy_registry.items():
            result = hybrid_manager.get_step(step_name, context)
            
            if result.step_definition:
                compatibility = self.check_step_compatibility(legacy_step, result.step_definition)
                
                if compatibility["compatible"]:
                    compatibility_report["compatible_steps"] += 1
                else:
                    compatibility_report["incompatible_steps"] += 1
                    compatibility_report["step_issues"][step_name] = compatibility
                    
                    # Check for migration blockers
                    if any("Missing required field" in issue for issue in compatibility["issues"]):
                        compatibility_report["migration_blockers"].append(step_name)
            else:
                compatibility_report["incompatible_steps"] += 1
                compatibility_report["step_issues"][step_name] = {
                    "compatible": False,
                    "issues": ["Step not found in hybrid registry"],
                    "warnings": [],
                    "legacy_fields": list(legacy_step.keys()),
                    "hybrid_fields": []
                }
                compatibility_report["migration_blockers"].append(step_name)
        
        # Generate overall warnings
        if compatibility_report["incompatible_steps"] > 0:
            compatibility_report["overall_warnings"].append(
                f"{compatibility_report['incompatible_steps']} steps have compatibility issues"
            )
        
        if compatibility_report["migration_blockers"]:
            compatibility_report["overall_warnings"].append(
                f"{len(compatibility_report['migration_blockers'])} steps are migration blockers"
            )
        
        return compatibility_report


class MigrationAssistant:
    """
    Assists with migration from legacy registry to hybrid registry.
    
    This assistant provides tools and utilities to help migrate existing
    code and configurations to the new hybrid registry system.
    """
    
    def __init__(self):
        self._converter = StepDefinitionConverter()
        self._compatibility_checker = APICompatibilityChecker()
        self._error_formatter = RegistryErrorFormatter()
    
    def generate_migration_plan(self, legacy_registry_path: str,
                              target_workspace_id: str = "migrated") -> Dict[str, Any]:
        """Generate a comprehensive migration plan."""
        
        migration_plan = {
            "source_registry": legacy_registry_path,
            "target_workspace": target_workspace_id,
            "migration_steps": [],
            "estimated_effort": "medium",
            "risk_assessment": "low",
            "prerequisites": [],
            "post_migration_tasks": []
        }
        
        # Add migration steps
        migration_plan["migration_steps"] = [
            "1. Backup existing registry files",
            "2. Initialize hybrid registry system",
            "3. Import legacy step definitions",
            "4. Validate step compatibility",
            "5. Update code to use hybrid API",
            "6. Test registry functionality",
            "7. Deploy hybrid registry"
        ]
        
        # Add prerequisites
        migration_plan["prerequisites"] = [
            "Ensure all step scripts are accessible",
            "Verify hyperparameter configurations",
            "Check dependency relationships",
            "Review workspace requirements"
        ]
        
        # Add post-migration tasks
        migration_plan["post_migration_tasks"] = [
            "Monitor performance metrics",
            "Update documentation",
            "Train team on hybrid API",
            "Remove deprecated code"
        ]
        
        return migration_plan
    
    def convert_legacy_registry(self, legacy_registry: Dict[str, Dict[str, Any]],
                              target_workspace_id: str) -> Dict[str, NamespacedStepDefinition]:
        """Convert legacy registry to hybrid format."""
        
        converted_steps = {}
        
        for step_name, legacy_step in legacy_registry.items():
            try:
                # Convert to base step definition
                base_step = self._converter.from_legacy_format(step_name, legacy_step)
                
                # Create namespaced step definition
                namespaced_step = NamespacedStepDefinition(
                    name=base_step.name,
                    step_type=base_step.step_type,
                    script_path=base_step.script_path,
                    hyperparameters=base_step.hyperparameters,
                    dependencies=base_step.dependencies,
                    metadata=base_step.metadata,
                    namespace=target_workspace_id,
                    workspace_id=target_workspace_id,
                    priority=1,
                    source_path="migrated_from_legacy"
                )
                
                converted_steps[step_name] = namespaced_step
                
            except Exception as e:
                logger.error(f"Failed to convert step {step_name}: {str(e)}")
        
        return converted_steps
    
    def validate_migration(self, legacy_registry: Dict[str, Dict[str, Any]],
                         hybrid_manager: HybridRegistryManager) -> Dict[str, Any]:
        """Validate that migration was successful."""
        
        return self._compatibility_checker.check_registry_compatibility(
            legacy_registry, hybrid_manager
        )
    
    def generate_migration_code(self, legacy_registry_path: str,
                              target_workspace_id: str) -> str:
        """Generate Python code for migration."""
        
        code_template = f'''
"""
Generated migration code for hybrid registry.
"""

from cursus.registry.hybrid import HybridRegistryManager, RegistryConfig

def migrate_to_hybrid_registry():
    """Migrate from legacy registry to hybrid registry."""
    
    # Initialize hybrid registry
    config = RegistryConfig(
        core_registry_path="{legacy_registry_path}",
        workspace_registry_paths=[],
        conflict_resolution_strategy="workspace_priority"
    )
    
    hybrid_manager = HybridRegistryManager(config)
    
    # Load all registries
    results = hybrid_manager.load_all_registries()
    
    # Validate loading
    for registry_name, result in results.items():
        if not result.is_valid:
            print(f"Warning: {{registry_name}} registry has issues: {{result.errors}}")
        else:
            print(f"Successfully loaded {{registry_name}} registry with {{result.step_count}} steps")
    
    return hybrid_manager

# Usage example
if __name__ == "__main__":
    manager = migrate_to_hybrid_registry()
    
    # Test step resolution
    from cursus.registry.hybrid.models import ResolutionContext
    
    context = ResolutionContext(workspace_id="{target_workspace_id}")
    result = manager.get_step("your_step_name", context)
    
    if result.step_definition:
        print(f"Successfully resolved step from {{result.source_registry}}")
    else:
        print(f"Failed to resolve step: {{result.errors}}")
'''
        
        return code_template.strip()


class BackwardCompatibilityValidator:
    """
    Validates backward compatibility across different versions.
    
    This validator ensures that changes to the hybrid registry system
    don't break existing functionality.
    """
    
    def __init__(self):
        self._compatibility_checker = APICompatibilityChecker()
    
    def validate_api_compatibility(self, legacy_interface: LegacyRegistryInterface,
                                 test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate API compatibility using test cases."""
        
        validation_results = {
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "compatibility_score": 0.0
        }
        
        for i, test_case in enumerate(test_cases):
            test_result = {
                "test_id": i,
                "test_name": test_case.get("name", f"test_{i}"),
                "passed": False,
                "errors": [],
                "warnings": []
            }
            
            try:
                # Execute test case
                method_name = test_case["method"]
                args = test_case.get("args", [])
                kwargs = test_case.get("kwargs", {})
                expected_result = test_case.get("expected_result")
                
                # Call legacy method
                method = getattr(legacy_interface, method_name)
                actual_result = method(*args, **kwargs)
                
                # Compare results
                if expected_result is not None:
                    if actual_result == expected_result:
                        test_result["passed"] = True
                        validation_results["passed_tests"] += 1
                    else:
                        test_result["errors"].append(f"Expected {expected_result}, got {actual_result}")
                        validation_results["failed_tests"] += 1
                else:
                    # If no expected result, just check that method doesn't crash
                    test_result["passed"] = True
                    validation_results["passed_tests"] += 1
                
            except Exception as e:
                test_result["errors"].append(str(e))
                validation_results["failed_tests"] += 1
            
            validation_results["test_results"].append(test_result)
        
        # Calculate compatibility score
        if validation_results["total_tests"] > 0:
            validation_results["compatibility_score"] = (
                validation_results["passed_tests"] / validation_results["total_tests"]
            )
        
        return validation_results
    
    def generate_compatibility_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable compatibility report."""
        
        report_lines = []
        report_lines.append("# Backward Compatibility Validation Report")
        report_lines.append("")
        
        # Summary
        score = validation_results["compatibility_score"]
        report_lines.append(f"## Summary")
        report_lines.append(f"- Compatibility Score: {score:.1%}")
        report_lines.append(f"- Total Tests: {validation_results['total_tests']}")
        report_lines.append(f"- Passed Tests: {validation_results['passed_tests']}")
        report_lines.append(f"- Failed Tests: {validation_results['failed_tests']}")
        report_lines.append("")
        
        # Test results
        if validation_results["test_results"]:
            report_lines.append("## Test Results")
            
            for test_result in validation_results["test_results"]:
                status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
                report_lines.append(f"### {test_result['test_name']} - {status}")
                
                if test_result["errors"]:
                    report_lines.append("**Errors:**")
                    for error in test_result["errors"]:
                        report_lines.append(f"- {error}")
                
                if test_result["warnings"]:
                    report_lines.append("**Warnings:**")
                    for warning in test_result["warnings"]:
                        report_lines.append(f"- {warning}")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        if score >= 0.9:
            report_lines.append("- ✅ High compatibility - safe to migrate")
        elif score >= 0.7:
            report_lines.append("- ⚠️ Good compatibility - review failed tests before migration")
        else:
            report_lines.append("- ❌ Low compatibility - significant issues need resolution")
        
        return "\n".join(report_lines)


# Global compatibility instance for easy access
_global_compatibility_layer: Optional[EnhancedBackwardCompatibilityLayer] = None


def get_compatibility_layer() -> Optional[EnhancedBackwardCompatibilityLayer]:
    """Get the global compatibility layer instance."""
    return _global_compatibility_layer


def set_compatibility_layer(layer: EnhancedBackwardCompatibilityLayer) -> None:
    """Set the global compatibility layer instance."""
    global _global_compatibility_layer
    _global_compatibility_layer = layer


def create_legacy_registry_interface(registry_path: str, workspace_id: str = "legacy") -> LegacyRegistryInterface:
    """Factory function to create a legacy registry interface."""
    interface = LegacyRegistryInterface(registry_path, workspace_id)
    
    # Set global compatibility layer
    set_compatibility_layer(interface.compatibility_layer)
    
    return interface
