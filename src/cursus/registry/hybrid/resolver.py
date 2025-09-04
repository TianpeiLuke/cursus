"""
Hybrid Registry Conflict Resolution Engine

This module provides advanced conflict resolution capabilities for the hybrid registry system,
including intelligent step resolution, dependency analysis, and conflict prevention strategies.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
from collections import defaultdict, deque
import hashlib
import json
from pydantic import BaseModel, Field

from .models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    StepResolutionResult,
    ConflictAnalysis,
    StepComponentResolution
)
from .utils import RegistryValidationUtils, RegistryErrorFormatter

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can occur in the registry."""
    NAME_COLLISION = "name_collision"
    DEPENDENCY_MISMATCH = "dependency_mismatch"
    VERSION_CONFLICT = "version_conflict"
    SCRIPT_PATH_CONFLICT = "script_path_conflict"
    HYPERPARAMETER_CONFLICT = "hyperparameter_conflict"
    CIRCULAR_DEPENDENCY = "circular_dependency"


class ResolutionStrategy(Enum):
    """Available resolution strategies."""
    WORKSPACE_PRIORITY = "workspace_priority"
    HIGHEST_PRIORITY = "highest_priority"
    CORE_FALLBACK = "core_fallback"
    MANUAL_RESOLUTION = "manual_resolution"
    FAIL_ON_CONFLICT = "fail_on_conflict"


class ConflictDetails(BaseModel):
    """Detailed information about a specific conflict."""
    conflict_type: ConflictType
    step_name: str
    conflicting_registries: List[str]
    conflicting_definitions: List[StepDefinition]
    severity: str = Field(default="medium", description="Severity level: low, medium, high, critical")
    resolution_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResolutionPlan(BaseModel):
    """Plan for resolving conflicts across the registry."""
    conflicts: List[ConflictDetails]
    resolution_order: List[str]
    automatic_resolutions: Dict[str, str]
    manual_resolutions: Dict[str, str]
    warnings: List[str]
    errors: List[str]


class DependencyAnalyzer:
    """Analyzes step dependencies and detects circular references."""
    
    def __init__(self):
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
    
    def analyze_dependencies(self, steps: Dict[str, StepDefinition]) -> Dict[str, List[str]]:
        """Analyze dependencies for all steps."""
        dependency_graph = {}
        
        for step_name, step_def in steps.items():
            dependencies = []
            if hasattr(step_def, 'dependencies') and step_def.dependencies:
                dependencies = step_def.dependencies
            dependency_graph[step_name] = dependencies
        
        return dependency_graph
    
    def detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the step graph."""
        def dfs(node: str, path: List[str], visited: Set[str], rec_stack: Set[str]) -> List[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            cycles = []
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    cycles.extend(dfs(neighbor, path.copy(), visited, rec_stack))
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
            return cycles
        
        all_cycles = []
        visited = set()
        
        for node in dependency_graph:
            if node not in visited:
                cycles = dfs(node, [], visited, set())
                all_cycles.extend(cycles)
        
        return all_cycles
    
    def get_dependency_order(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Get topological order of steps based on dependencies."""
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for node in dependency_graph:
            in_degree[node] = 0
        
        for node, deps in dependency_graph.items():
            for dep in deps:
                in_degree[dep] += 1
        
        # Queue for nodes with no incoming edges
        queue = deque([node for node in dependency_graph if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Remove this node from the graph
            for dep in dependency_graph.get(node, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        
        return result


class ConflictDetector:
    """Detects various types of conflicts in the registry."""
    
    def __init__(self):
        self._dependency_analyzer = DependencyAnalyzer()
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
    
    def detect_all_conflicts(self, core_steps: Dict[str, StepDefinition],
                           local_steps: Dict[str, Dict[str, NamespacedStepDefinition]]) -> List[ConflictDetails]:
        """Detect all types of conflicts across registries."""
        conflicts = []
        
        # Combine all steps for analysis
        all_steps = {}
        step_sources = {}
        
        # Add core steps
        for name, step in core_steps.items():
            all_steps[name] = step
            step_sources[name] = ["core"]
        
        # Add local steps
        for workspace_id, workspace_steps in local_steps.items():
            for name, step in workspace_steps.items():
                if name in all_steps:
                    step_sources[name].append(workspace_id)
                else:
                    step_sources[name] = [workspace_id]
                all_steps[name] = step
        
        # Detect name collisions
        conflicts.extend(self._detect_name_collisions(step_sources, all_steps))
        
        # Detect dependency conflicts
        conflicts.extend(self._detect_dependency_conflicts(all_steps))
        
        # Detect circular dependencies
        conflicts.extend(self._detect_circular_dependencies(all_steps))
        
        # Detect script path conflicts
        conflicts.extend(self._detect_script_path_conflicts(all_steps, step_sources))
        
        # Detect hyperparameter conflicts
        conflicts.extend(self._detect_hyperparameter_conflicts(all_steps, step_sources))
        
        return conflicts
    
    def _detect_name_collisions(self, step_sources: Dict[str, List[str]], 
                               all_steps: Dict[str, StepDefinition]) -> List[ConflictDetails]:
        """Detect name collisions between registries."""
        conflicts = []
        
        for step_name, sources in step_sources.items():
            if len(sources) > 1:
                conflicting_definitions = []
                for source in sources:
                    # This is simplified - in practice we'd need to track which definition comes from which source
                    conflicting_definitions.append(all_steps[step_name])
                
                conflict = ConflictDetails(
                    conflict_type=ConflictType.NAME_COLLISION,
                    step_name=step_name,
                    conflicting_registries=sources,
                    conflicting_definitions=conflicting_definitions,
                    severity="medium",
                    resolution_suggestion=f"Use workspace priority or rename step in local registry",
                    metadata={"collision_count": len(sources)}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_dependency_conflicts(self, all_steps: Dict[str, StepDefinition]) -> List[ConflictDetails]:
        """Detect dependency-related conflicts."""
        conflicts = []
        
        for step_name, step_def in all_steps.items():
            if hasattr(step_def, 'dependencies') and step_def.dependencies:
                for dep in step_def.dependencies:
                    if dep not in all_steps:
                        conflict = ConflictDetails(
                            conflict_type=ConflictType.DEPENDENCY_MISMATCH,
                            step_name=step_name,
                            conflicting_registries=["missing"],
                            conflicting_definitions=[step_def],
                            severity="high",
                            resolution_suggestion=f"Add missing dependency '{dep}' or remove from dependencies",
                            metadata={"missing_dependency": dep}
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_circular_dependencies(self, all_steps: Dict[str, StepDefinition]) -> List[ConflictDetails]:
        """Detect circular dependencies."""
        conflicts = []
        
        dependency_graph = self._dependency_analyzer.analyze_dependencies(all_steps)
        cycles = self._dependency_analyzer.detect_circular_dependencies(dependency_graph)
        
        for cycle in cycles:
            conflict = ConflictDetails(
                conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                step_name=cycle[0],  # Use first step in cycle as primary
                conflicting_registries=["dependency_cycle"],
                conflicting_definitions=[all_steps[step] for step in cycle if step in all_steps],
                severity="critical",
                resolution_suggestion=f"Break circular dependency: {' -> '.join(cycle)}",
                metadata={"cycle": cycle, "cycle_length": len(cycle)}
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_script_path_conflicts(self, all_steps: Dict[str, StepDefinition],
                                    step_sources: Dict[str, List[str]]) -> List[ConflictDetails]:
        """Detect script path conflicts."""
        conflicts = []
        script_paths = defaultdict(list)
        
        # Group steps by script path
        for step_name, step_def in all_steps.items():
            if hasattr(step_def, 'script_path') and step_def.script_path:
                script_paths[step_def.script_path].append(step_name)
        
        # Check for conflicts
        for script_path, step_names in script_paths.items():
            if len(step_names) > 1:
                # Multiple steps using same script path
                conflicting_definitions = [all_steps[name] for name in step_names]
                all_sources = []
                for name in step_names:
                    all_sources.extend(step_sources.get(name, []))
                
                conflict = ConflictDetails(
                    conflict_type=ConflictType.SCRIPT_PATH_CONFLICT,
                    step_name=step_names[0],  # Use first as primary
                    conflicting_registries=list(set(all_sources)),
                    conflicting_definitions=conflicting_definitions,
                    severity="low",
                    resolution_suggestion=f"Steps sharing script path: {script_path}. Ensure this is intentional.",
                    metadata={"shared_script_path": script_path, "affected_steps": step_names}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_hyperparameter_conflicts(self, all_steps: Dict[str, StepDefinition],
                                       step_sources: Dict[str, List[str]]) -> List[ConflictDetails]:
        """Detect hyperparameter conflicts between step definitions."""
        conflicts = []
        
        # Group steps by name to check for hyperparameter differences
        for step_name, sources in step_sources.items():
            if len(sources) > 1:
                # Get all definitions for this step name
                definitions = []
                for source in sources:
                    # In practice, we'd need to track which definition comes from which source
                    definitions.append(all_steps[step_name])
                
                # Compare hyperparameters
                if len(set(str(getattr(d, 'hyperparameters', {})) for d in definitions)) > 1:
                    conflict = ConflictDetails(
                        conflict_type=ConflictType.HYPERPARAMETER_CONFLICT,
                        step_name=step_name,
                        conflicting_registries=sources,
                        conflicting_definitions=definitions,
                        severity="medium",
                        resolution_suggestion="Review hyperparameter differences and choose appropriate version",
                        metadata={"hyperparameter_variations": len(definitions)}
                    )
                    conflicts.append(conflict)
        
        return conflicts


class ConflictResolver:
    """Resolves conflicts between step definitions using various strategies."""
    
    def __init__(self):
        self._detector = ConflictDetector()
        self._dependency_analyzer = DependencyAnalyzer()
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
    
    def create_resolution_plan(self, core_steps: Dict[str, StepDefinition],
                             local_steps: Dict[str, Dict[str, NamespacedStepDefinition]],
                             default_strategy: ResolutionStrategy = ResolutionStrategy.WORKSPACE_PRIORITY) -> ResolutionPlan:
        """Create a comprehensive resolution plan for all conflicts."""
        
        # Detect all conflicts
        conflicts = self._detector.detect_all_conflicts(core_steps, local_steps)
        
        # Sort conflicts by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        conflicts.sort(key=lambda c: severity_order.get(c.severity, 3))
        
        # Create resolution plan
        automatic_resolutions = {}
        manual_resolutions = {}
        warnings = []
        errors = []
        
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.CIRCULAR_DEPENDENCY:
                # Circular dependencies require manual resolution
                manual_resolutions[conflict.step_name] = conflict.resolution_suggestion or "Manual intervention required"
                errors.append(f"Critical: Circular dependency detected for {conflict.step_name}")
                
            elif conflict.conflict_type == ConflictType.NAME_COLLISION:
                # Name collisions can often be resolved automatically
                if default_strategy == ResolutionStrategy.WORKSPACE_PRIORITY:
                    automatic_resolutions[conflict.step_name] = "Use workspace-specific version"
                elif default_strategy == ResolutionStrategy.CORE_FALLBACK:
                    automatic_resolutions[conflict.step_name] = "Use core registry version"
                else:
                    automatic_resolutions[conflict.step_name] = "Use highest priority version"
                warnings.append(f"Name collision resolved for {conflict.step_name}")
                
            elif conflict.conflict_type == ConflictType.DEPENDENCY_MISMATCH:
                # Missing dependencies are errors
                errors.append(f"Missing dependency for {conflict.step_name}: {conflict.metadata.get('missing_dependency')}")
                manual_resolutions[conflict.step_name] = "Add missing dependency or update step definition"
                
            else:
                # Other conflicts get warnings and automatic resolution
                automatic_resolutions[conflict.step_name] = conflict.resolution_suggestion or "Use default resolution"
                warnings.append(f"{conflict.conflict_type.value} for {conflict.step_name}")
        
        # Determine resolution order based on dependencies
        all_step_names = set()
        all_step_names.update(core_steps.keys())
        for workspace_steps in local_steps.values():
            all_step_names.update(workspace_steps.keys())
        
        # Create combined dependency graph for ordering
        combined_steps = dict(core_steps)
        for workspace_steps in local_steps.values():
            combined_steps.update(workspace_steps)
        
        dependency_graph = self._dependency_analyzer.analyze_dependencies(combined_steps)
        resolution_order = self._dependency_analyzer.get_dependency_order(dependency_graph)
        
        return ResolutionPlan(
            conflicts=conflicts,
            resolution_order=resolution_order,
            automatic_resolutions=automatic_resolutions,
            manual_resolutions=manual_resolutions,
            warnings=warnings,
            errors=errors
        )
    
    def resolve_step_with_strategy(self, step_name: str, candidates: List[Tuple[StepDefinition, str, int]],
                                 strategy: ResolutionStrategy, context: ResolutionContext) -> StepResolutionResult:
        """Resolve a specific step using the given strategy."""
        
        if not candidates:
            error_msg = self._error_formatter.format_step_not_found_error(step_name, [])
            return StepResolutionResult(
                step_definition=None,
                source_registry="none",
                workspace_id=context.workspace_id,
                resolution_strategy=strategy.value,
                conflict_detected=False,
                conflict_analysis=None,
                errors=[error_msg],
                warnings=[]
            )
        
        # Single candidate - no conflict
        if len(candidates) == 1:
            step_def, source, priority = candidates[0]
            return StepResolutionResult(
                step_definition=step_def,
                source_registry=source,
                workspace_id=context.workspace_id,
                resolution_strategy=strategy.value,
                conflict_detected=False,
                conflict_analysis=None,
                errors=[],
                warnings=[]
            )
        
        # Multiple candidates - apply resolution strategy
        selected = self._apply_resolution_strategy(candidates, strategy, context)
        step_def, source, priority = selected
        
        # Create conflict analysis
        conflict_analysis = ConflictAnalysis(
            step_name=step_name,
            conflicting_sources=[source for _, source, _ in candidates],
            resolution_strategy=strategy.value,
            workspace_context=context.workspace_id
        )
        
        # Generate warnings
        warnings = []
        other_sources = [c[1] for c in candidates if c[1] != source]
        if other_sources:
            warnings.append(f"Step '{step_name}' found in multiple registries: {other_sources}. Using {source}.")
        
        return StepResolutionResult(
            step_definition=step_def,
            source_registry=source,
            workspace_id=context.workspace_id,
            resolution_strategy=strategy.value,
            conflict_detected=True,
            conflict_analysis=conflict_analysis,
            errors=[],
            warnings=warnings
        )
    
    def _apply_resolution_strategy(self, candidates: List[Tuple[StepDefinition, str, int]],
                                 strategy: ResolutionStrategy, context: ResolutionContext) -> Tuple[StepDefinition, str, int]:
        """Apply the specified resolution strategy to select from candidates."""
        
        if strategy == ResolutionStrategy.WORKSPACE_PRIORITY:
            # Prefer workspace-specific steps, then by priority
            workspace_candidates = [c for c in candidates if c[1] == context.workspace_id]
            if workspace_candidates:
                return max(workspace_candidates, key=lambda x: x[2])
            else:
                return max(candidates, key=lambda x: x[2])
                
        elif strategy == ResolutionStrategy.HIGHEST_PRIORITY:
            # Select by highest priority
            return max(candidates, key=lambda x: x[2])
            
        elif strategy == ResolutionStrategy.CORE_FALLBACK:
            # Prefer core registry
            core_candidates = [c for c in candidates if c[1] == "core"]
            if core_candidates:
                return core_candidates[0]
            else:
                return max(candidates, key=lambda x: x[2])
                
        elif strategy == ResolutionStrategy.FAIL_ON_CONFLICT:
            # Fail if there are conflicts
            raise ValueError(f"Conflict resolution failed: multiple candidates found")
            
        else:
            # Default to highest priority
            return max(candidates, key=lambda x: x[2])


class StepResolver:
    """
    Advanced step resolver with intelligent conflict resolution.
    
    This resolver provides sophisticated step resolution capabilities including
    dependency analysis, conflict detection, and resolution strategy application.
    """
    
    def __init__(self):
        self._conflict_detector = ConflictDetector()
        self._conflict_resolver = ConflictResolver()
        self._dependency_analyzer = DependencyAnalyzer()
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
    
    def resolve_step_with_dependencies(self, step_name: str, 
                                     core_steps: Dict[str, StepDefinition],
                                     local_steps: Dict[str, Dict[str, NamespacedStepDefinition]],
                                     context: ResolutionContext) -> StepComponentResolution:
        """
        Resolve a step and all its dependencies.
        
        Returns a comprehensive resolution result including the main step
        and all resolved dependencies.
        """
        
        resolved_steps = {}
        resolution_order = []
        conflicts = []
        warnings = []
        errors = []
        
        # Get all available steps
        all_steps = dict(core_steps)
        for workspace_steps in local_steps.values():
            all_steps.update(workspace_steps)
        
        # Build dependency graph starting from the requested step
        to_resolve = deque([step_name])
        resolved = set()
        
        while to_resolve:
            current_step = to_resolve.popleft()
            
            if current_step in resolved:
                continue
                
            if current_step not in all_steps:
                errors.append(f"Step not found: {current_step}")
                continue
            
            # Collect candidates for this step
            candidates = []
            
            # Check core registry
            if current_step in core_steps:
                candidates.append((core_steps[current_step], "core", 0))
            
            # Check local registries
            for workspace_id, workspace_steps in local_steps.items():
                if current_step in workspace_steps:
                    step_def = workspace_steps[current_step]
                    priority = getattr(step_def, 'priority', 1)
                    candidates.append((step_def, workspace_id, priority))
            
            # Resolve this step
            if candidates:
                try:
                    strategy = ResolutionStrategy(context.resolution_strategy)
                    resolution_result = self._conflict_resolver.resolve_step_with_strategy(
                        current_step, candidates, strategy, context
                    )
                    
                    if resolution_result.step_definition:
                        resolved_steps[current_step] = resolution_result
                        resolution_order.append(current_step)
                        resolved.add(current_step)
                        
                        if resolution_result.conflict_detected:
                            conflicts.append(resolution_result.conflict_analysis)
                        
                        warnings.extend(resolution_result.warnings)
                        errors.extend(resolution_result.errors)
                        
                        # Add dependencies to resolution queue
                        step_def = resolution_result.step_definition
                        if hasattr(step_def, 'dependencies') and step_def.dependencies:
                            for dep in step_def.dependencies:
                                if dep not in resolved:
                                    to_resolve.append(dep)
                    else:
                        errors.extend(resolution_result.errors)
                        
                except Exception as e:
                    error_msg = f"Failed to resolve step {current_step}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            else:
                errors.append(f"No candidates found for step: {current_step}")
        
        return StepComponentResolution(
            primary_step=step_name,
            resolved_steps=resolved_steps,
            resolution_order=resolution_order,
            conflicts=conflicts,
            warnings=warnings,
            errors=errors,
            context=context
        )
    
    def validate_resolution_plan(self, plan: ResolutionPlan) -> bool:
        """Validate that a resolution plan is feasible."""
        
        # Check for critical errors
        critical_conflicts = [c for c in plan.conflicts if c.severity == "critical"]
        if critical_conflicts and not plan.manual_resolutions:
            return False
        
        # Check for unresolved dependencies
        for conflict in plan.conflicts:
            if conflict.conflict_type == ConflictType.DEPENDENCY_MISMATCH:
                if conflict.step_name not in plan.manual_resolutions:
                    return False
        
        return True
    
    def generate_resolution_report(self, plan: ResolutionPlan) -> str:
        """Generate a human-readable resolution report."""
        
        report_lines = []
        report_lines.append("# Registry Conflict Resolution Report")
        report_lines.append("")
        
        # Summary
        report_lines.append(f"## Summary")
        report_lines.append(f"- Total conflicts detected: {len(plan.conflicts)}")
        report_lines.append(f"- Automatic resolutions: {len(plan.automatic_resolutions)}")
        report_lines.append(f"- Manual resolutions required: {len(plan.manual_resolutions)}")
        report_lines.append(f"- Warnings: {len(plan.warnings)}")
        report_lines.append(f"- Errors: {len(plan.errors)}")
        report_lines.append("")
        
        # Conflicts by type
        conflict_by_type = defaultdict(list)
        for conflict in plan.conflicts:
            conflict_by_type[conflict.conflict_type].append(conflict)
        
        report_lines.append("## Conflicts by Type")
        for conflict_type, conflicts in conflict_by_type.items():
            report_lines.append(f"### {conflict_type.value.replace('_', ' ').title()}")
            for conflict in conflicts:
                report_lines.append(f"- **{conflict.step_name}**: {conflict.resolution_suggestion}")
                report_lines.append(f"  - Severity: {conflict.severity}")
                report_lines.append(f"  - Registries: {', '.join(conflict.conflicting_registries)}")
            report_lines.append("")
        
        # Resolution actions
        if plan.automatic_resolutions:
            report_lines.append("## Automatic Resolutions")
            for step_name, resolution in plan.automatic_resolutions.items():
                report_lines.append(f"- **{step_name}**: {resolution}")
            report_lines.append("")
        
        if plan.manual_resolutions:
            report_lines.append("## Manual Resolutions Required")
            for step_name, resolution in plan.manual_resolutions.items():
                report_lines.append(f"- **{step_name}**: {resolution}")
            report_lines.append("")
        
        # Warnings and errors
        if plan.warnings:
            report_lines.append("## Warnings")
            for warning in plan.warnings:
                report_lines.append(f"- {warning}")
            report_lines.append("")
        
        if plan.errors:
            report_lines.append("## Errors")
            for error in plan.errors:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        return "\n".join(report_lines)


class AdvancedStepResolver:
    """
    Advanced step resolver with caching and optimization.
    
    This resolver provides enhanced capabilities including resolution caching,
    performance optimization, and advanced conflict prevention.
    """
    
    def __init__(self, enable_caching: bool = True, cache_size: int = 1000):
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._resolution_cache: Dict[str, StepResolutionResult] = {}
        self._dependency_cache: Dict[str, List[str]] = {}
        self._conflict_cache: Dict[str, List[ConflictDetails]] = {}
        
        self._step_resolver = StepResolver()
        self._conflict_resolver = ConflictResolver()
        self._dependency_analyzer = DependencyAnalyzer()
    
    def resolve_with_caching(self, step_name: str, 
                           core_steps: Dict[str, StepDefinition],
                           local_steps: Dict[str, Dict[str, NamespacedStepDefinition]],
                           context: ResolutionContext) -> StepResolutionResult:
        """Resolve step with caching for improved performance."""
        
        if not self.enable_caching:
            return self._resolve_without_cache(step_name, core_steps, local_steps, context)
        
        # Generate cache key
        cache_key = self._generate_cache_key(step_name, core_steps, local_steps, context)
        
        # Check cache
        if cache_key in self._resolution_cache:
            logger.debug(f"Cache hit for step resolution: {step_name}")
            return self._resolution_cache[cache_key]
        
        # Resolve and cache
        result = self._resolve_without_cache(step_name, core_steps, local_steps, context)
        
        # Manage cache size
        if len(self._resolution_cache) >= self.cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._resolution_cache))
            del self._resolution_cache[oldest_key]
        
        self._resolution_cache[cache_key] = result
        logger.debug(f"Cached step resolution: {step_name}")
        
        return result
    
    def _resolve_without_cache(self, step_name: str,
                             core_steps: Dict[str, StepDefinition],
                             local_steps: Dict[str, Dict[str, NamespacedStepDefinition]],
                             context: ResolutionContext) -> StepResolutionResult:
        """Resolve step without caching."""
        
        # Collect candidates
        candidates = []
        
        # Check core registry
        if step_name in core_steps:
            candidates.append((core_steps[step_name], "core", 0))
        
        # Check local registries
        for workspace_id, workspace_steps in local_steps.items():
            if step_name in workspace_steps:
                step_def = workspace_steps[step_name]
                priority = getattr(step_def, 'priority', 1)
                candidates.append((step_def, workspace_id, priority))
        
        # Use conflict resolver
        try:
            strategy = ResolutionStrategy(context.resolution_strategy)
            return self._conflict_resolver.resolve_step_with_strategy(
                step_name, candidates, strategy, context
            )
        except Exception as e:
            error_msg = f"Resolution failed for {step_name}: {str(e)}"
            return StepResolutionResult(
                step_definition=None,
                source_registry="error",
                workspace_id=context.workspace_id,
                resolution_strategy=context.resolution_strategy,
                conflict_detected=True,
                conflict_analysis=None,
                errors=[error_msg],
                warnings=[]
            )
    
    def _generate_cache_key(self, step_name: str,
                          core_steps: Dict[str, StepDefinition],
                          local_steps: Dict[str, Dict[str, NamespacedStepDefinition]],
                          context: ResolutionContext) -> str:
        """Generate a cache key for the resolution request."""
        
        # Create a hash of the relevant state
        key_data = {
            "step_name": step_name,
            "workspace_id": context.workspace_id,
            "resolution_strategy": context.resolution_strategy,
            "core_steps_hash": self._hash_steps(core_steps),
            "local_steps_hash": {ws_id: self._hash_steps(ws_steps) for ws_id, ws_steps in local_steps.items()}
        }
        
        # Generate hash
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()
    
    def _hash_steps(self, steps: Dict[str, StepDefinition]) -> str:
        """Generate a hash for a collection of steps."""
        step_data = {}
        for name, step in steps.items():
            step_data[name] = {
                "step_type": step.step_type,
                "script_path": step.script_path,
                "hyperparameters": step.hyperparameters,
                "dependencies": step.dependencies
            }
        
        step_json = json.dumps(step_data, sort_keys=True)
        return hashlib.md5(step_json.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear all cached resolutions."""
        self._resolution_cache.clear()
        self._dependency_cache.clear()
        self._conflict_cache.clear()
        logger.info("Cleared resolution cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "resolution_cache_size": len(self._resolution_cache),
            "dependency_cache_size": len(self._dependency_cache),
            "conflict_cache_size": len(self._conflict_cache),
            "max_cache_size": self.cache_size
        }
