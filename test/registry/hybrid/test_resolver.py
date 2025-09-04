"""
Test suite for hybrid registry conflict resolution components.

Tests ConflictDetector, ConflictResolver, StepResolver, and AdvancedStepResolver.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.cursus.registry.hybrid.resolver import (
    ConflictType,
    ResolutionStrategy,
    ConflictDetails,
    ResolutionPlan,
    DependencyAnalyzer,
    ConflictDetector,
    ConflictResolver,
    StepResolver,
    AdvancedStepResolver
)
from src.cursus.registry.hybrid.models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    StepResolutionResult,
    ConflictAnalysis,
    StepComponentResolution
)


class TestDependencyAnalyzer(unittest.TestCase):
    """Test DependencyAnalyzer component."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = DependencyAnalyzer()
        
        # Create mock step definitions with dependencies
        self.step_a = Mock(spec=StepDefinition)
        self.step_a.dependencies = ["step_b", "step_c"]
        
        self.step_b = Mock(spec=StepDefinition)
        self.step_b.dependencies = ["step_c"]
        
        self.step_c = Mock(spec=StepDefinition)
        self.step_c.dependencies = []
        
        self.step_d = Mock(spec=StepDefinition)
        self.step_d.dependencies = ["step_a"]  # Creates cycle with step_a -> step_b -> step_c
        
        self.steps = {
            "step_a": self.step_a,
            "step_b": self.step_b,
            "step_c": self.step_c,
            "step_d": self.step_d
        }
    
    def test_analyze_dependencies(self):
        """Test dependency analysis."""
        dependency_graph = self.analyzer.analyze_dependencies(self.steps)
        
        self.assertEqual(dependency_graph["step_a"], ["step_b", "step_c"])
        self.assertEqual(dependency_graph["step_b"], ["step_c"])
        self.assertEqual(dependency_graph["step_c"], [])
        self.assertEqual(dependency_graph["step_d"], ["step_a"])
    
    def test_detect_circular_dependencies_no_cycles(self):
        """Test circular dependency detection with no cycles."""
        # Remove step_d to eliminate cycle
        steps_no_cycle = {k: v for k, v in self.steps.items() if k != "step_d"}
        dependency_graph = self.analyzer.analyze_dependencies(steps_no_cycle)
        
        cycles = self.analyzer.detect_circular_dependencies(dependency_graph)
        self.assertEqual(len(cycles), 0)
    
    def test_detect_circular_dependencies_with_cycles(self):
        """Test circular dependency detection with cycles."""
        # Create a cycle: step_a -> step_b -> step_a
        self.step_b.dependencies = ["step_a"]
        
        dependency_graph = self.analyzer.analyze_dependencies(self.steps)
        cycles = self.analyzer.detect_circular_dependencies(dependency_graph)
        
        self.assertGreater(len(cycles), 0)
        # Should detect the cycle involving step_a and step_b
        cycle_steps = set()
        for cycle in cycles:
            cycle_steps.update(cycle)
        self.assertIn("step_a", cycle_steps)
        self.assertIn("step_b", cycle_steps)
    
    def test_get_dependency_order(self):
        """Test topological ordering of dependencies."""
        # Remove step_d to avoid cycles
        steps_no_cycle = {k: v for k, v in self.steps.items() if k != "step_d"}
        dependency_graph = self.analyzer.analyze_dependencies(steps_no_cycle)
        
        order = self.analyzer.get_dependency_order(dependency_graph)
        
        # step_c should come before step_b, step_b before step_a
        c_index = order.index("step_c")
        b_index = order.index("step_b")
        a_index = order.index("step_a")
        
        self.assertLess(c_index, b_index)
        self.assertLess(b_index, a_index)


class TestConflictDetector(unittest.TestCase):
    """Test ConflictDetector component."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = ConflictDetector()
        
        # Create mock step definitions
        self.core_step = Mock(spec=StepDefinition)
        self.core_step.step_type = "Processing"
        self.core_step.script_path = "core_script.py"
        self.core_step.hyperparameters = {"param1": "value1"}
        self.core_step.dependencies = []
        
        self.local_step = Mock(spec=NamespacedStepDefinition)
        self.local_step.step_type = "Processing"
        self.local_step.script_path = "local_script.py"
        self.local_step.hyperparameters = {"param1": "value2"}
        self.local_step.dependencies = []
        
        self.core_steps = {"TestStep": self.core_step}
        self.local_steps = {"workspace1": {"TestStep": self.local_step}}
    
    def test_detect_name_collisions(self):
        """Test name collision detection."""
        conflicts = self.detector.detect_all_conflicts(self.core_steps, self.local_steps)
        
        # Should detect name collision for TestStep
        name_collisions = [c for c in conflicts if c.conflict_type == ConflictType.NAME_COLLISION]
        self.assertEqual(len(name_collisions), 1)
        self.assertEqual(name_collisions[0].step_name, "TestStep")
        self.assertIn("core", name_collisions[0].conflicting_registries)
        self.assertIn("workspace1", name_collisions[0].conflicting_registries)
    
    def test_detect_dependency_conflicts(self):
        """Test dependency conflict detection."""
        # Add missing dependency
        self.core_step.dependencies = ["MissingStep"]
        
        conflicts = self.detector.detect_all_conflicts(self.core_steps, self.local_steps)
        
        # Should detect dependency mismatch
        dep_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY_MISMATCH]
        self.assertEqual(len(dep_conflicts), 1)
        self.assertEqual(dep_conflicts[0].step_name, "TestStep")
        self.assertEqual(dep_conflicts[0].metadata["missing_dependency"], "MissingStep")
    
    def test_detect_script_path_conflicts(self):
        """Test script path conflict detection."""
        # Make both steps use same script path
        self.local_step.script_path = "core_script.py"
        
        conflicts = self.detector.detect_all_conflicts(self.core_steps, self.local_steps)
        
        # Should detect script path conflict
        script_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.SCRIPT_PATH_CONFLICT]
        self.assertEqual(len(script_conflicts), 1)
        self.assertEqual(script_conflicts[0].metadata["shared_script_path"], "core_script.py")
    
    def test_detect_hyperparameter_conflicts(self):
        """Test hyperparameter conflict detection."""
        conflicts = self.detector.detect_all_conflicts(self.core_steps, self.local_steps)
        
        # Should detect hyperparameter conflict (different param1 values)
        hyper_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.HYPERPARAMETER_CONFLICT]
        self.assertEqual(len(hyper_conflicts), 1)
        self.assertEqual(hyper_conflicts[0].step_name, "TestStep")


class TestConflictResolver(unittest.TestCase):
    """Test ConflictResolver component."""
    
    def setUp(self):
        """Set up test environment."""
        self.resolver = ConflictResolver()
        
        # Create mock step definitions
        self.core_step = Mock(spec=StepDefinition)
        self.core_step.step_type = "Processing"
        self.core_step.dependencies = []
        
        self.local_step = Mock(spec=NamespacedStepDefinition)
        self.local_step.step_type = "Processing"
        self.local_step.dependencies = []
        self.local_step.priority = 90
        
        self.core_steps = {"TestStep": self.core_step}
        self.local_steps = {"workspace1": {"TestStep": self.local_step}}
        
        self.context = ResolutionContext(
            workspace_id="workspace1",
            resolution_strategy="workspace_priority"
        )
    
    def test_create_resolution_plan(self):
        """Test resolution plan creation."""
        plan = self.resolver.create_resolution_plan(
            self.core_steps, 
            self.local_steps,
            ResolutionStrategy.WORKSPACE_PRIORITY
        )
        
        self.assertIsInstance(plan, ResolutionPlan)
        self.assertGreater(len(plan.conflicts), 0)  # Should detect name collision
        self.assertIn("TestStep", plan.automatic_resolutions)
    
    def test_resolve_step_with_workspace_priority(self):
        """Test step resolution with workspace priority strategy."""
        candidates = [
            (self.core_step, "core", 0),
            (self.local_step, "workspace1", 90)
        ]
        
        result = self.resolver.resolve_step_with_strategy(
            "TestStep", 
            candidates, 
            ResolutionStrategy.WORKSPACE_PRIORITY,
            self.context
        )
        
        self.assertIsNotNone(result.step_definition)
        self.assertEqual(result.source_registry, "workspace1")
        self.assertTrue(result.conflict_detected)
        self.assertEqual(result.resolution_strategy, "workspace_priority")
    
    def test_resolve_step_with_highest_priority(self):
        """Test step resolution with highest priority strategy."""
        candidates = [
            (self.core_step, "core", 0),
            (self.local_step, "workspace1", 90)
        ]
        
        result = self.resolver.resolve_step_with_strategy(
            "TestStep",
            candidates,
            ResolutionStrategy.HIGHEST_PRIORITY,
            self.context
        )
        
        self.assertIsNotNone(result.step_definition)
        self.assertEqual(result.source_registry, "workspace1")  # Higher priority
        self.assertTrue(result.conflict_detected)
    
    def test_resolve_step_with_core_fallback(self):
        """Test step resolution with core fallback strategy."""
        candidates = [
            (self.core_step, "core", 0),
            (self.local_step, "workspace1", 90)
        ]
        
        result = self.resolver.resolve_step_with_strategy(
            "TestStep",
            candidates,
            ResolutionStrategy.CORE_FALLBACK,
            self.context
        )
        
        self.assertIsNotNone(result.step_definition)
        self.assertEqual(result.source_registry, "core")  # Core fallback
        self.assertTrue(result.conflict_detected)
    
    def test_resolve_step_no_candidates(self):
        """Test step resolution with no candidates."""
        result = self.resolver.resolve_step_with_strategy(
            "NonExistentStep",
            [],
            ResolutionStrategy.WORKSPACE_PRIORITY,
            self.context
        )
        
        self.assertIsNone(result.step_definition)
        self.assertEqual(result.source_registry, "none")
        self.assertFalse(result.conflict_detected)
        self.assertGreater(len(result.errors), 0)
    
    def test_resolve_step_single_candidate(self):
        """Test step resolution with single candidate."""
        candidates = [(self.core_step, "core", 0)]
        
        result = self.resolver.resolve_step_with_strategy(
            "TestStep",
            candidates,
            ResolutionStrategy.WORKSPACE_PRIORITY,
            self.context
        )
        
        self.assertIsNotNone(result.step_definition)
        self.assertEqual(result.source_registry, "core")
        self.assertFalse(result.conflict_detected)
        self.assertEqual(len(result.errors), 0)


class TestStepResolver(unittest.TestCase):
    """Test StepResolver component."""
    
    def setUp(self):
        """Set up test environment."""
        self.resolver = StepResolver()
        
        # Create mock step definitions with dependencies
        self.step_a = Mock(spec=StepDefinition)
        self.step_a.step_type = "Processing"
        self.step_a.dependencies = ["step_b"]
        
        self.step_b = Mock(spec=StepDefinition)
        self.step_b.step_type = "Processing"
        self.step_b.dependencies = []
        
        self.core_steps = {
            "step_a": self.step_a,
            "step_b": self.step_b
        }
        self.local_steps = {}
        
        self.context = ResolutionContext(
            workspace_id=None,
            resolution_strategy="highest_priority"
        )
    
    def test_resolve_step_with_dependencies(self):
        """Test resolving step with dependencies."""
        result = self.resolver.resolve_step_with_dependencies(
            "step_a",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        self.assertIsInstance(result, StepComponentResolution)
        self.assertEqual(result.primary_step, "step_a")
        self.assertIn("step_a", result.resolved_steps)
        self.assertIn("step_b", result.resolved_steps)  # Dependency should be resolved
        self.assertEqual(len(result.errors), 0)
    
    def test_resolve_step_missing_dependency(self):
        """Test resolving step with missing dependency."""
        # Add missing dependency
        self.step_a.dependencies = ["missing_step"]
        
        result = self.resolver.resolve_step_with_dependencies(
            "step_a",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        self.assertEqual(result.primary_step, "step_a")
        self.assertGreater(len(result.errors), 0)
        # Should have error about missing dependency
        error_messages = " ".join(result.errors)
        self.assertIn("missing_step", error_messages)
    
    def test_resolve_nonexistent_step(self):
        """Test resolving non-existent step."""
        result = self.resolver.resolve_step_with_dependencies(
            "nonexistent_step",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        self.assertEqual(result.primary_step, "nonexistent_step")
        self.assertEqual(len(result.resolved_steps), 0)
        self.assertGreater(len(result.errors), 0)


class TestAdvancedStepResolver(unittest.TestCase):
    """Test AdvancedStepResolver component."""
    
    def setUp(self):
        """Set up test environment."""
        self.resolver = AdvancedStepResolver(enable_caching=True, cache_size=10)
        
        # Create mock step definition
        self.core_step = Mock(spec=StepDefinition)
        self.core_step.step_type = "Processing"
        self.core_step.script_path = "test_script.py"
        self.core_step.hyperparameters = {"param1": "value1"}
        self.core_step.dependencies = []
        
        self.core_steps = {"TestStep": self.core_step}
        self.local_steps = {}
        
        self.context = ResolutionContext(
            workspace_id=None,
            resolution_strategy="highest_priority"
        )
    
    def test_resolve_with_caching_enabled(self):
        """Test resolution with caching enabled."""
        # First resolution - should cache
        result1 = self.resolver.resolve_with_caching(
            "TestStep",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        # Second resolution - should use cache
        result2 = self.resolver.resolve_with_caching(
            "TestStep",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        self.assertIsNotNone(result1.step_definition)
        self.assertIsNotNone(result2.step_definition)
        self.assertEqual(result1.source_registry, result2.source_registry)
        
        # Check cache stats
        stats = self.resolver.get_cache_stats()
        self.assertEqual(stats["resolution_cache_size"], 1)
    
    def test_resolve_with_caching_disabled(self):
        """Test resolution with caching disabled."""
        resolver_no_cache = AdvancedStepResolver(enable_caching=False)
        
        result = resolver_no_cache.resolve_with_caching(
            "TestStep",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        self.assertIsNotNone(result.step_definition)
        
        # Cache should be empty
        stats = resolver_no_cache.get_cache_stats()
        self.assertEqual(stats["resolution_cache_size"], 0)
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        small_resolver = AdvancedStepResolver(enable_caching=True, cache_size=2)
        
        # Create multiple steps to exceed cache size
        steps = {}
        for i in range(5):
            step = Mock(spec=StepDefinition)
            step.step_type = "Processing"
            step.script_path = f"script_{i}.py"
            step.hyperparameters = {}
            step.dependencies = []
            steps[f"Step{i}"] = step
        
        # Resolve all steps
        for step_name in steps:
            small_resolver.resolve_with_caching(
                step_name,
                {step_name: steps[step_name]},
                {},
                self.context
            )
        
        # Cache should not exceed limit
        stats = small_resolver.get_cache_stats()
        self.assertLessEqual(stats["resolution_cache_size"], 2)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.resolver.resolve_with_caching(
            "TestStep",
            self.core_steps,
            self.local_steps,
            self.context
        )
        
        # Verify cache has content
        stats_before = self.resolver.get_cache_stats()
        self.assertGreater(stats_before["resolution_cache_size"], 0)
        
        # Clear cache
        self.resolver.clear_cache()
        
        # Verify cache is empty
        stats_after = self.resolver.get_cache_stats()
        self.assertEqual(stats_after["resolution_cache_size"], 0)
    
    def test_cache_key_generation(self):
        """Test cache key generation for different contexts."""
        context1 = ResolutionContext(workspace_id="workspace1", resolution_strategy="workspace_priority")
        context2 = ResolutionContext(workspace_id="workspace2", resolution_strategy="workspace_priority")
        
        # Resolve with different contexts
        result1 = self.resolver.resolve_with_caching("TestStep", self.core_steps, self.local_steps, context1)
        result2 = self.resolver.resolve_with_caching("TestStep", self.core_steps, self.local_steps, context2)
        
        # Should have separate cache entries
        stats = self.resolver.get_cache_stats()
        self.assertEqual(stats["resolution_cache_size"], 2)


if __name__ == '__main__':
    unittest.main()
