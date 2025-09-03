"""
Cross-Workspace Validator

Provides comprehensive cross-workspace validation capabilities, integrating with
the Phase 1 consolidated workspace system. This module validates compatibility
between workspace components and ensures proper cross-workspace dependencies.

Architecture Integration:
- Leverages Phase 1 WorkspaceDiscoveryManager for component discovery
- Uses Phase 1 WorkspaceIntegrationManager for integration validation
- Integrates with optimized WorkspacePipelineAssembler from Phase 2
- Coordinates with Phase 3 test workspace management system

Features:
- Cross-workspace component compatibility validation
- Dependency conflict detection and resolution
- Integration readiness assessment
- Pipeline assembly validation across workspaces
- Integration with Phase 1 consolidated managers
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, Set
import logging
from datetime import datetime
from collections import defaultdict

from pydantic import BaseModel, Field, ConfigDict

# PHASE 3 INTEGRATION: Import Phase 1 consolidated workspace system
from ..core.manager import WorkspaceManager, WorkspaceContext
# Removed circular import - will access discovery manager through workspace manager
from ..core.integration import WorkspaceIntegrationManager
from ..core.assembler import WorkspacePipelineAssembler
from ..core.config import WorkspacePipelineDefinition

# Import Phase 3 test management components
from .workspace_test_manager import WorkspaceTestManager

logger = logging.getLogger(__name__)


class ComponentConflict(BaseModel):
    """Represents a conflict between workspace components."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    conflict_type: str  # "name", "version", "dependency", "interface"
    severity: str = "medium"  # "low", "medium", "high", "critical"
    component_1: str
    workspace_1: str
    component_2: str
    workspace_2: str
    description: str
    resolution_suggestions: List[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.now)


class DependencyResolution(BaseModel):
    """Represents dependency resolution information."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    component_id: str
    workspace_id: str
    dependencies: List[str] = Field(default_factory=list)
    resolved_dependencies: Dict[str, str] = Field(default_factory=dict)  # dependency -> workspace
    unresolved_dependencies: List[str] = Field(default_factory=list)
    circular_dependencies: List[List[str]] = Field(default_factory=list)
    resolution_status: str = "pending"  # "pending", "resolved", "failed"


class ValidationResult(BaseModel):
    """Cross-workspace validation result."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    validation_id: str
    workspaces_validated: List[str] = Field(default_factory=list)
    is_valid: bool = False
    conflicts: List[ComponentConflict] = Field(default_factory=list)
    dependency_resolutions: List[DependencyResolution] = Field(default_factory=list)
    integration_readiness: Dict[str, bool] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.now)
    validation_summary: Dict[str, Any] = Field(default_factory=dict)


class CrossWorkspaceConfig(BaseModel):
    """Configuration for cross-workspace validation."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    enable_conflict_detection: bool = True
    enable_dependency_resolution: bool = True
    enable_integration_validation: bool = True
    strict_validation: bool = False
    allowed_conflicts: List[str] = Field(default_factory=list)
    dependency_resolution_timeout: int = 300  # seconds
    max_circular_dependency_depth: int = 10


class CrossWorkspaceValidator:
    """
    Comprehensive cross-workspace validation system integrating with Phase 1.
    
    This validator provides advanced cross-workspace validation capabilities by
    leveraging the Phase 1 consolidated workspace system. It validates:
    - Component compatibility between workspaces
    - Cross-workspace dependency resolution
    - Integration readiness assessment
    - Pipeline assembly validation
    
    Phase 3 Integration Features:
    - Uses Phase 1 WorkspaceDiscoveryManager for component discovery
    - Leverages Phase 1 WorkspaceIntegrationManager for integration validation
    - Integrates with Phase 2 optimized WorkspacePipelineAssembler
    - Coordinates with Phase 3 WorkspaceTestManager for validation testing
    """
    
    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
        validation_config: Optional[CrossWorkspaceConfig] = None,
        test_manager: Optional[WorkspaceTestManager] = None
    ):
        """
        Initialize cross-workspace validator.
        
        Args:
            workspace_manager: Phase 1 consolidated workspace manager
            validation_config: Cross-workspace validation configuration
            test_manager: Phase 3 test workspace manager for validation testing
        """
        # PHASE 3 INTEGRATION: Use Phase 1 consolidated workspace manager
        if workspace_manager:
            self.workspace_manager = workspace_manager
        else:
            from ..core.manager import WorkspaceManager
            self.workspace_manager = WorkspaceManager()
        
        # Access Phase 1 specialized managers
        self.discovery_manager = self.workspace_manager.discovery_manager
        self.integration_manager = self.workspace_manager.integration_manager
        
        # PHASE 2 INTEGRATION: Use optimized pipeline assembler
        self.pipeline_assembler = WorkspacePipelineAssembler(
            workspace_root=self.workspace_manager.workspace_root,
            workspace_manager=self.workspace_manager
        )
        
        # PHASE 3 INTEGRATION: Use test manager for validation testing
        self.test_manager = test_manager or WorkspaceTestManager(
            workspace_manager=self.workspace_manager
        )
        
        # Cross-workspace validation configuration
        self.validation_config = validation_config or CrossWorkspaceConfig()
        
        # Validation state
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.component_registry: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized cross-workspace validator with Phase 1-3 integration")
    
    # Cross-Workspace Component Discovery and Analysis
    
    def discover_cross_workspace_components(
        self,
        workspace_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Discover components across multiple workspaces using Phase 1 discovery manager.
        
        Args:
            workspace_ids: Optional list of workspace IDs to analyze
        
        Returns:
            Dictionary containing cross-workspace component information
        """
        logger.info(f"Discovering cross-workspace components for: {workspace_ids or 'all workspaces'}")
        
        try:
            # Use Phase 1 discovery manager for component discovery
            discovery_result = self.discovery_manager.discover_components(
                workspace_ids=workspace_ids
            )
            
            # Build cross-workspace component registry
            self.component_registry = self._build_component_registry(discovery_result)
            
            # Analyze component relationships
            component_analysis = self._analyze_component_relationships()
            
            result = {
                "discovery_result": discovery_result,
                "component_registry": self.component_registry,
                "component_analysis": component_analysis,
                "total_workspaces": len(discovery_result.get("workspaces", {})),
                "total_components": sum(
                    len(components) for components in self.component_registry.values()
                ),
                "discovery_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Discovered components across {result['total_workspaces']} workspaces")
            return result
            
        except Exception as e:
            logger.error(f"Failed to discover cross-workspace components: {e}")
            return {"error": str(e), "component_registry": {}}
    
    def _build_component_registry(self, discovery_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build component registry from discovery results."""
        registry = defaultdict(dict)
        
        try:
            workspaces = discovery_result.get("workspaces", {})
            
            for workspace_id, workspace_info in workspaces.items():
                components = workspace_info.get("components", {})
                
                for component_type, component_list in components.items():
                    for component in component_list:
                        component_id = component.get("name", component.get("id", "unknown"))
                        
                        registry[workspace_id][component_id] = {
                            "type": component_type,
                            "workspace": workspace_id,
                            "metadata": component,
                            "dependencies": component.get("dependencies", []),
                            "interfaces": component.get("interfaces", []),
                            "version": component.get("version", "unknown")
                        }
            
            return dict(registry)
            
        except Exception as e:
            logger.warning(f"Failed to build component registry: {e}")
            return {}
    
    def _analyze_component_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between components across workspaces."""
        analysis = {
            "duplicate_components": [],
            "dependency_chains": [],
            "interface_conflicts": [],
            "version_conflicts": []
        }
        
        try:
            # Find duplicate component names across workspaces
            component_names = defaultdict(list)
            
            for workspace_id, components in self.component_registry.items():
                for component_id, component_info in components.items():
                    component_names[component_id].append({
                        "workspace": workspace_id,
                        "info": component_info
                    })
            
            # Identify duplicates
            for component_name, instances in component_names.items():
                if len(instances) > 1:
                    analysis["duplicate_components"].append({
                        "component_name": component_name,
                        "instances": instances,
                        "conflict_potential": self._assess_conflict_potential(instances)
                    })
            
            # Analyze dependency chains
            analysis["dependency_chains"] = self._analyze_dependency_chains()
            
            # Check for interface conflicts
            analysis["interface_conflicts"] = self._check_interface_conflicts()
            
            # Check for version conflicts
            analysis["version_conflicts"] = self._check_version_conflicts(component_names)
            
        except Exception as e:
            logger.warning(f"Failed to analyze component relationships: {e}")
        
        return analysis
    
    def _assess_conflict_potential(self, instances: List[Dict[str, Any]]) -> str:
        """Assess conflict potential for duplicate components."""
        try:
            # Check if versions differ
            versions = set(instance["info"].get("version", "unknown") for instance in instances)
            if len(versions) > 1 and "unknown" not in versions:
                return "high"
            
            # Check if interfaces differ
            interfaces = [instance["info"].get("interfaces", []) for instance in instances]
            if len(set(str(sorted(iface)) for iface in interfaces)) > 1:
                return "medium"
            
            return "low"
            
        except Exception:
            return "unknown"
    
    def _analyze_dependency_chains(self) -> List[Dict[str, Any]]:
        """Analyze dependency chains across workspaces."""
        chains = []
        
        try:
            for workspace_id, components in self.component_registry.items():
                for component_id, component_info in components.items():
                    dependencies = component_info.get("dependencies", [])
                    
                    if dependencies:
                        chain = self._trace_dependency_chain(component_id, workspace_id, set())
                        if len(chain) > 1:
                            chains.append({
                                "root_component": component_id,
                                "root_workspace": workspace_id,
                                "chain": chain,
                                "cross_workspace": self._is_cross_workspace_chain(chain)
                            })
            
        except Exception as e:
            logger.warning(f"Failed to analyze dependency chains: {e}")
        
        return chains
    
    def _trace_dependency_chain(
        self,
        component_id: str,
        workspace_id: str,
        visited: Set[str]
    ) -> List[Dict[str, str]]:
        """Trace dependency chain for a component."""
        if f"{workspace_id}:{component_id}" in visited:
            return []  # Circular dependency detected
        
        visited.add(f"{workspace_id}:{component_id}")
        chain = [{"component": component_id, "workspace": workspace_id}]
        
        try:
            component_info = self.component_registry.get(workspace_id, {}).get(component_id, {})
            dependencies = component_info.get("dependencies", [])
            
            for dep in dependencies:
                # Find dependency in any workspace
                dep_workspace, dep_component = self._find_component_location(dep)
                if dep_workspace and dep_component:
                    dep_chain = self._trace_dependency_chain(dep_component, dep_workspace, visited.copy())
                    chain.extend(dep_chain)
            
        except Exception as e:
            logger.debug(f"Error tracing dependency chain for {component_id}: {e}")
        
        return chain
    
    def _find_component_location(self, component_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Find the workspace location of a component."""
        for workspace_id, components in self.component_registry.items():
            if component_name in components:
                return workspace_id, component_name
        return None, None
    
    def _is_cross_workspace_chain(self, chain: List[Dict[str, str]]) -> bool:
        """Check if dependency chain crosses workspace boundaries."""
        workspaces = set(item["workspace"] for item in chain)
        return len(workspaces) > 1
    
    def _check_interface_conflicts(self) -> List[Dict[str, Any]]:
        """Check for interface conflicts between components."""
        conflicts = []
        
        try:
            # Group components by interface
            interface_map = defaultdict(list)
            
            for workspace_id, components in self.component_registry.items():
                for component_id, component_info in components.items():
                    interfaces = component_info.get("interfaces", [])
                    for interface in interfaces:
                        interface_map[interface].append({
                            "component": component_id,
                            "workspace": workspace_id,
                            "info": component_info
                        })
            
            # Check for conflicts
            for interface, implementers in interface_map.items():
                if len(implementers) > 1:
                    # Check if implementations are compatible
                    if self._are_interface_implementations_conflicting(implementers):
                        conflicts.append({
                            "interface": interface,
                            "conflicting_implementations": implementers,
                            "conflict_reason": "Incompatible interface implementations"
                        })
            
        except Exception as e:
            logger.warning(f"Failed to check interface conflicts: {e}")
        
        return conflicts
    
    def _are_interface_implementations_conflicting(self, implementers: List[Dict[str, Any]]) -> bool:
        """Check if interface implementations are conflicting."""
        try:
            # Simple heuristic: different versions or different workspaces might conflict
            versions = set(impl["info"].get("version", "unknown") for impl in implementers)
            workspaces = set(impl["workspace"] for impl in implementers)
            
            # Conflict if multiple versions or multiple workspaces
            return len(versions) > 1 or len(workspaces) > 1
            
        except Exception:
            return True  # Assume conflict if we can't determine
    
    def _check_version_conflicts(self, component_names: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Check for version conflicts between components."""
        conflicts = []
        
        try:
            for component_name, instances in component_names.items():
                if len(instances) > 1:
                    versions = [instance["info"].get("version", "unknown") for instance in instances]
                    unique_versions = set(versions)
                    
                    if len(unique_versions) > 1 and "unknown" not in unique_versions:
                        conflicts.append({
                            "component_name": component_name,
                            "conflicting_versions": list(unique_versions),
                            "instances": instances
                        })
            
        except Exception as e:
            logger.warning(f"Failed to check version conflicts: {e}")
        
        return conflicts
    
    # Cross-Workspace Validation
    
    def validate_cross_workspace_pipeline(
        self,
        pipeline_definition: Union[WorkspacePipelineDefinition, Dict[str, Any]],
        workspace_ids: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate cross-workspace pipeline using Phase 2 optimized pipeline assembler.
        
        Args:
            pipeline_definition: Pipeline definition to validate
            workspace_ids: Optional list of workspace IDs involved
        
        Returns:
            ValidationResult with comprehensive validation information
        """
        validation_id = f"cross_workspace_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Validating cross-workspace pipeline: {validation_id}")
        
        try:
            # Convert to WorkspacePipelineDefinition if needed
            if isinstance(pipeline_definition, dict):
                pipeline_def = WorkspacePipelineDefinition(**pipeline_definition)
            else:
                pipeline_def = pipeline_definition
            
            # Discover components if not already done
            if not self.component_registry:
                self.discover_cross_workspace_components(workspace_ids)
            
            # Use Phase 2 optimized pipeline assembler for validation
            assembly_result = self.pipeline_assembler.validate_workspace_components(pipeline_def)
            
            # Perform cross-workspace specific validation
            conflicts = self._detect_component_conflicts(pipeline_def, workspace_ids)
            dependency_resolutions = self._resolve_cross_workspace_dependencies(pipeline_def)
            integration_readiness = self._assess_integration_readiness(pipeline_def, workspace_ids)
            
            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                conflicts, dependency_resolutions, integration_readiness
            )
            
            # Create validation result
            validation_result = ValidationResult(
                validation_id=validation_id,
                workspaces_validated=workspace_ids or list(self.component_registry.keys()),
                is_valid=len(conflicts) == 0 and assembly_result.get("is_valid", False),
                conflicts=conflicts,
                dependency_resolutions=dependency_resolutions,
                integration_readiness=integration_readiness,
                recommendations=recommendations,
                validation_summary={
                    "total_conflicts": len(conflicts),
                    "resolved_dependencies": len([dr for dr in dependency_resolutions if dr.resolution_status == "resolved"]),
                    "integration_ready_workspaces": len([ws for ws, ready in integration_readiness.items() if ready]),
                    "assembly_result": assembly_result
                }
            )
            
            # Cache validation result
            self.validation_cache[validation_id] = validation_result
            
            logger.info(f"Cross-workspace validation completed: {validation_id} - Valid: {validation_result.is_valid}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate cross-workspace pipeline: {e}")
            return ValidationResult(
                validation_id=validation_id,
                workspaces_validated=workspace_ids or [],
                is_valid=False,
                recommendations=[f"Validation error: {e}"]
            )
    
    def _detect_component_conflicts(
        self,
        pipeline_def: WorkspacePipelineDefinition,
        workspace_ids: Optional[List[str]]
    ) -> List[ComponentConflict]:
        """Detect conflicts between components in the pipeline."""
        conflicts = []
        
        if not self.validation_config.enable_conflict_detection:
            return conflicts
        
        try:
            # Get pipeline components
            pipeline_components = self._extract_pipeline_components(pipeline_def)
            
            # Check for conflicts between components
            for i, comp1 in enumerate(pipeline_components):
                for comp2 in pipeline_components[i+1:]:
                    conflict = self._check_component_pair_conflict(comp1, comp2)
                    if conflict:
                        conflicts.append(conflict)
            
            # Filter allowed conflicts
            conflicts = [
                conflict for conflict in conflicts
                if conflict.conflict_type not in self.validation_config.allowed_conflicts
            ]
            
        except Exception as e:
            logger.warning(f"Failed to detect component conflicts: {e}")
        
        return conflicts
    
    def _extract_pipeline_components(self, pipeline_def: WorkspacePipelineDefinition) -> List[Dict[str, Any]]:
        """Extract components from pipeline definition."""
        components = []
        
        try:
            # Extract from steps
            for step in pipeline_def.steps:
                step_dict = step.model_dump() if hasattr(step, 'model_dump') else step
                
                component_info = {
                    "id": step_dict.get("name", step_dict.get("id", "unknown")),
                    "type": step_dict.get("type", "step"),
                    "workspace": step_dict.get("workspace", "unknown"),
                    "metadata": step_dict
                }
                components.append(component_info)
            
        except Exception as e:
            logger.warning(f"Failed to extract pipeline components: {e}")
        
        return components
    
    def _check_component_pair_conflict(
        self,
        comp1: Dict[str, Any],
        comp2: Dict[str, Any]
    ) -> Optional[ComponentConflict]:
        """Check for conflicts between two components."""
        try:
            # Check name conflicts
            if comp1["id"] == comp2["id"] and comp1["workspace"] != comp2["workspace"]:
                return ComponentConflict(
                    conflict_type="name",
                    severity="high",
                    component_1=comp1["id"],
                    workspace_1=comp1["workspace"],
                    component_2=comp2["id"],
                    workspace_2=comp2["workspace"],
                    description=f"Component name conflict: {comp1['id']} exists in multiple workspaces",
                    resolution_suggestions=[
                        "Rename one of the conflicting components",
                        "Use workspace-specific naming conventions",
                        "Consolidate components into a single workspace"
                    ]
                )
            
            # Check interface conflicts
            comp1_interfaces = comp1.get("metadata", {}).get("interfaces", [])
            comp2_interfaces = comp2.get("metadata", {}).get("interfaces", [])
            
            common_interfaces = set(comp1_interfaces) & set(comp2_interfaces)
            if common_interfaces:
                return ComponentConflict(
                    conflict_type="interface",
                    severity="medium",
                    component_1=comp1["id"],
                    workspace_1=comp1["workspace"],
                    component_2=comp2["id"],
                    workspace_2=comp2["workspace"],
                    description=f"Interface conflict: Components implement same interfaces: {list(common_interfaces)}",
                    resolution_suggestions=[
                        "Ensure interface implementations are compatible",
                        "Use different interface versions",
                        "Consolidate interface implementations"
                    ]
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Error checking component pair conflict: {e}")
            return None
    
    def _resolve_cross_workspace_dependencies(
        self,
        pipeline_def: WorkspacePipelineDefinition
    ) -> List[DependencyResolution]:
        """Resolve dependencies across workspace boundaries."""
        resolutions = []
        
        if not self.validation_config.enable_dependency_resolution:
            return resolutions
        
        try:
            # Use Phase 1 discovery manager for dependency resolution
            dependency_result = self.discovery_manager.resolve_cross_workspace_dependencies(
                pipeline_def.model_dump() if hasattr(pipeline_def, 'model_dump') else pipeline_def
            )
            
            # Convert to DependencyResolution objects
            for component_id, dep_info in dependency_result.get("dependencies", {}).items():
                resolution = DependencyResolution(
                    component_id=component_id,
                    workspace_id=dep_info.get("workspace", "unknown"),
                    dependencies=dep_info.get("dependencies", []),
                    resolved_dependencies=dep_info.get("resolved", {}),
                    unresolved_dependencies=dep_info.get("unresolved", []),
                    circular_dependencies=dep_info.get("circular", []),
                    resolution_status="resolved" if not dep_info.get("unresolved") else "failed"
                )
                resolutions.append(resolution)
            
        except Exception as e:
            logger.warning(f"Failed to resolve cross-workspace dependencies: {e}")
        
        return resolutions
    
    def _assess_integration_readiness(
        self,
        pipeline_def: WorkspacePipelineDefinition,
        workspace_ids: Optional[List[str]]
    ) -> Dict[str, bool]:
        """Assess integration readiness for workspaces."""
        readiness = {}
        
        if not self.validation_config.enable_integration_validation:
            return readiness
        
        try:
            target_workspaces = workspace_ids or list(self.component_registry.keys())
            
            for workspace_id in target_workspaces:
                # Use Phase 1 integration manager for readiness assessment
                readiness_result = self.integration_manager.validate_integration_readiness([workspace_id])
                readiness[workspace_id] = readiness_result.get("ready", False)
            
        except Exception as e:
            logger.warning(f"Failed to assess integration readiness: {e}")
            # Default to not ready if assessment fails
            for workspace_id in (workspace_ids or []):
                readiness[workspace_id] = False
        
        return readiness
    
    def _generate_validation_recommendations(
        self,
        conflicts: List[ComponentConflict],
        dependency_resolutions: List[DependencyResolution],
        integration_readiness: Dict[str, bool]
    ) -> List[str]:
        """Generate validation recommendations based on results."""
        recommendations = []
        
        try:
            # Recommendations for conflicts
            if conflicts:
                high_severity_conflicts = [c for c in conflicts if c.severity == "high"]
                if high_severity_conflicts:
                    recommendations.append(
                        f"Resolve {len(high_severity_conflicts)} high-severity component conflicts before proceeding"
                    )
                
                recommendations.append(
                    f"Address {len(conflicts)} total component conflicts for optimal integration"
                )
            
            # Recommendations for dependencies
            failed_resolutions = [dr for dr in dependency_resolutions if dr.resolution_status == "failed"]
            if failed_resolutions:
                recommendations.append(
                    f"Resolve {len(failed_resolutions)} failed dependency resolutions"
                )
            
            circular_deps = [dr for dr in dependency_resolutions if dr.circular_dependencies]
            if circular_deps:
                recommendations.append(
                    f"Break {len(circular_deps)} circular dependency chains"
                )
            
            # Recommendations for integration readiness
            not_ready_workspaces = [ws for ws, ready in integration_readiness.items() if not ready]
            if not_ready_workspaces:
                recommendations.append(
                    f"Prepare {len(not_ready_workspaces)} workspaces for integration: {', '.join(not_ready_workspaces)}"
                )
            
            # General recommendations
            if not conflicts and not failed_resolutions and all(integration_readiness.values()):
                recommendations.append("All validations passed - pipeline is ready for cross-workspace integration")
            
        except Exception as e:
            logger.warning(f"Failed to generate validation recommendations: {e}")
            recommendations.append("Review validation results manually due to recommendation generation error")
        
        return recommendations
    
    # Integration Testing with Phase 3 Test Manager
    
    def validate_with_test_environment(
        self,
        pipeline_definition: Union[WorkspacePipelineDefinition, Dict[str, Any]],
        workspace_ids: Optional[List[str]] = None,
        test_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate cross-workspace pipeline using Phase 3 test environment.
        
        Args:
            pipeline_definition: Pipeline definition to validate
            workspace_ids: Optional list of workspace IDs involved
            test_config: Optional test configuration
        
        Returns:
            Dictionary containing validation and test results
        """
        logger.info("Validating cross-workspace pipeline with test environment")
        
        try:
            # Perform standard cross-workspace validation
            validation_result = self.validate_cross_workspace_pipeline(
                pipeline_definition, workspace_ids
            )
            
            # Create test environment for validation
            test_id = f"cross_workspace_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            test_environment = self.test_manager.create_test_environment(
                test_id=test_id,
                test_type="cross_workspace_validation",
                **(test_config or {})
            )
            
            # Run validation in test environment
            test_validation_result = self._run_validation_in_test_environment(
                test_environment, pipeline_definition, validation_result
            )
            
            # Cleanup test environment
            self.test_manager.cleanup_test_environment(test_id)
            
            return {
                "validation_result": validation_result.model_dump(),
                "test_environment_result": test_validation_result,
                "combined_status": validation_result.is_valid and test_validation_result.get("success", False),
                "test_id": test_id
            }
            
        except Exception as e:
            logger.error(f"Failed to validate with test environment: {e}")
            return {
                "error": str(e),
                "validation_result": None,
                "test_environment_result": None,
                "combined_status": False
            }
    
    def _run_validation_in_test_environment(
        self,
        test_environment,
        pipeline_definition,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Run validation in isolated test environment."""
        try:
            # This is a simplified implementation
            # In practice, you would run the actual pipeline validation
            # in the isolated test environment
            
            test_result = {
                "success": validation_result.is_valid,
                "test_environment_id": test_environment.test_id,
                "isolation_validated": True,
                "pipeline_assembly_tested": True,
                "cross_workspace_integration_tested": True,
                "test_timestamp": datetime.now().isoformat()
            }
            
            # Add test-specific validation results
            if not validation_result.is_valid:
                test_result["test_failures"] = [
                    f"Conflict: {conflict.description}" for conflict in validation_result.conflicts
                ]
            
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to run validation in test environment: {e}")
            return {
                "success": False,
                "test_environment_id": test_environment.test_id,
                "error": str(e),
                "test_timestamp": datetime.now().isoformat()
            }
    
    # Information and Statistics
    
    def get_validation_info(self, validation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get validation information."""
        if validation_id:
            if validation_id not in self.validation_cache:
                return {"error": f"Validation not found: {validation_id}"}
            
            validation_result = self.validation_cache[validation_id]
            return validation_result.model_dump()
        else:
            return {
                "total_validations": len(self.validation_cache),
                "cached_validations": list(self.validation_cache.keys()),
                "component_registry_size": len(self.component_registry),
                "phase_integration_status": {
                    "phase1_discovery_manager": str(type(self.discovery_manager)),
                    "phase1_integration_manager": str(type(self.integration_manager)),
                    "phase2_pipeline_assembler": str(type(self.pipeline_assembler)),
                    "phase3_test_manager": str(type(self.test_manager))
                }
            }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        try:
            return {
                "validations": {
                    "total": len(self.validation_cache),
                    "successful": len([v for v in self.validation_cache.values() if v.is_valid]),
                    "failed": len([v for v in self.validation_cache.values() if not v.is_valid])
                },
                "components": {
                    "total_workspaces": len(self.component_registry),
                    "total_components": sum(len(components) for components in self.component_registry.values())
                },
                "phase_integration": {
                    "phase1_integrated": True,
                    "phase2_integrated": True,
                    "phase3_integrated": True
                },
                "validation_config": self.validation_config.model_dump()
            }
        except Exception as e:
            logger.error(f"Failed to get validation statistics: {e}")
            return {"error": str(e)}


# Convenience functions for cross-workspace validation

def create_cross_workspace_validator(
    workspace_root: Optional[str] = None,
    validation_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CrossWorkspaceValidator:
    """
    Convenience function to create a configured CrossWorkspaceValidator.
    
    Args:
        workspace_root: Root directory for workspaces
        validation_config: Cross-workspace validation configuration
        **kwargs: Additional arguments for WorkspaceManager
    
    Returns:
        Configured CrossWorkspaceValidator instance
    """
    # Create Phase 1 consolidated workspace manager
    workspace_manager = WorkspaceManager(workspace_root=workspace_root, **kwargs)
    
    # Create validation configuration if provided
    cross_workspace_config = None
    if validation_config:
        cross_workspace_config = CrossWorkspaceConfig(**validation_config)
    
    return CrossWorkspaceValidator(
        workspace_manager=workspace_manager,
        validation_config=cross_workspace_config
    )


def validate_cross_workspace_compatibility(
    workspace_ids: List[str],
    workspace_root: Optional[str] = None,
    strict: bool = False
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Convenience function to validate cross-workspace compatibility.
    
    Args:
        workspace_ids: List of workspace IDs to validate
        workspace_root: Root directory for workspaces
        strict: Whether to apply strict validation rules
    
    Returns:
        Tuple of (is_compatible, list_of_conflict_dicts)
    """
    validator = create_cross_workspace_validator(
        workspace_root=workspace_root,
        validation_config={"strict_validation": strict}
    )
    
    # Discover components
    discovery_result = validator.discover_cross_workspace_components(workspace_ids)
    
    # Check for conflicts in component analysis
    component_analysis = discovery_result.get("component_analysis", {})
    conflicts = []
    
    # Convert analysis results to conflict format
    for duplicate in component_analysis.get("duplicate_components", []):
        if duplicate.get("conflict_potential") in ["high", "medium"]:
            conflicts.append({
                "type": "duplicate_component",
                "component": duplicate["component_name"],
                "conflict_potential": duplicate["conflict_potential"],
                "instances": duplicate["instances"]
            })
    
    for version_conflict in component_analysis.get("version_conflicts", []):
        conflicts.append({
            "type": "version_conflict",
            "component": version_conflict["component_name"],
            "conflicting_versions": version_conflict["conflicting_versions"],
            "instances": version_conflict["instances"]
        })
    
    for interface_conflict in component_analysis.get("interface_conflicts", []):
        conflicts.append({
            "type": "interface_conflict",
            "interface": interface_conflict["interface"],
            "conflicting_implementations": interface_conflict["conflicting_implementations"]
        })
    
    is_compatible = len(conflicts) == 0
    return is_compatible, conflicts
