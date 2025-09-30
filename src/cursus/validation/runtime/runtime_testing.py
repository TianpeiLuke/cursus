"""
Simplified Pipeline Runtime Testing

Validates script functionality and data transfer consistency for pipeline development.
Based on validated user story: "examine the script's functionality and their data 
transfer consistency along the DAG, without worrying about the resolution of 
step-to-step or step-to-script dependencies."

Refactored implementation with PipelineTestingSpecBuilder and ScriptExecutionSpec
for user-centric approach with local persistence.
"""

import importlib.util
import json
import os
import time
import argparse
import pandas as pd
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Import from separate model files
from .runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration,
)
from .runtime_spec_builder import PipelineTestingSpecBuilder

# Import inference testing models
from .runtime_inference import (
    InferenceHandlerSpec,
    InferenceTestResult,
    InferencePipelineTestingSpec,
)

# Import PipelineDAG for integration
from ...api.dag.base_dag import PipelineDAG

# Import Phase 2 logical name matching components (optional)
try:
    from .logical_name_matching import (
        PathMatcher,
        TopologicalExecutor,
        LogicalNameMatchingTester,
        EnhancedScriptExecutionSpec,
        EnhancedDataCompatibilityResult,
    )

    LOGICAL_MATCHING_AVAILABLE = True
except ImportError:
    LOGICAL_MATCHING_AVAILABLE = False
    PathMatcher = None
    TopologicalExecutor = None
    LogicalNameMatchingTester = None
    EnhancedScriptExecutionSpec = None
    EnhancedDataCompatibilityResult = None


class RuntimeTester:
    """Core testing engine that uses PipelineTestingSpecBuilder for parameter extraction"""

    def __init__(
        self,
        config_or_workspace_dir,
        enable_logical_matching: bool = True,
        semantic_threshold: float = 0.7,
        step_catalog: Optional['StepCatalog'] = None,
    ):
        # Support both new RuntimeTestingConfiguration and old string workspace_dir for backward compatibility
        if isinstance(config_or_workspace_dir, RuntimeTestingConfiguration):
            self.config = config_or_workspace_dir
            self.pipeline_spec = config_or_workspace_dir.pipeline_spec
            self.workspace_dir = Path(
                config_or_workspace_dir.pipeline_spec.test_workspace_root
            )

            # Create builder instance for parameter extraction
            self.builder = PipelineTestingSpecBuilder(
                test_data_dir=config_or_workspace_dir.pipeline_spec.test_workspace_root
            )
        else:
            # Backward compatibility: treat as workspace directory string
            workspace_dir = str(config_or_workspace_dir)
            self.config = None
            self.pipeline_spec = None
            self.workspace_dir = Path(workspace_dir)

            # Create builder instance for parameter extraction
            self.builder = PipelineTestingSpecBuilder(test_data_dir=workspace_dir)

        # Initialize Phase 2 logical name matching components (optional)
        self.enable_logical_matching = (
            enable_logical_matching and LOGICAL_MATCHING_AVAILABLE
        )
        if self.enable_logical_matching:
            self.path_matcher = PathMatcher(semantic_threshold)
            self.topological_executor = TopologicalExecutor()
            self.logical_name_tester = LogicalNameMatchingTester(semantic_threshold)
        else:
            self.path_matcher = None
            self.topological_executor = None
            self.logical_name_tester = None

        # NEW: Step Catalog Integration (Phase 1)
        self.step_catalog = step_catalog or self._initialize_step_catalog()

    # Phase 1: Step Catalog Integration Methods

    def _initialize_step_catalog(self):
        """
        Initialize step catalog with unified workspace resolution.
        
        Resolves the conflict between RuntimeTester's config_or_workspace_dir and 
        StepCatalog's workspace_dirs by creating a unified workspace discovery strategy.
        
        Priority order:
        1. test_data_dir (primary testing workspace)
        2. RuntimeTester's workspace_dir (secondary testing workspace)
        3. Additional development workspaces from environment
        4. Package-only discovery (for deployment scenarios)
        """
        try:
            from ...step_catalog import StepCatalog
        except ImportError:
            # Step catalog not available, return None for optional enhancement
            return None
        
        workspace_dirs = []
        
        # Priority 1: Use test_data_dir as primary workspace
        if hasattr(self.builder, 'test_data_dir') and self.builder.test_data_dir:
            test_workspace = Path(self.builder.test_data_dir) / "scripts"
            if test_workspace.exists():
                workspace_dirs.append(test_workspace)
            else:
                test_data_path = Path(self.builder.test_data_dir)
                if test_data_path.exists():
                    workspace_dirs.append(test_data_path)
        
        # Priority 2: Add RuntimeTester's workspace_dir if different
        if hasattr(self, 'workspace_dir') and self.workspace_dir:
            runtime_workspace = Path(self.workspace_dir)
            if runtime_workspace not in workspace_dirs and runtime_workspace.exists():
                workspace_dirs.append(runtime_workspace)
        
        # Priority 3: Add development workspaces from environment
        dev_workspaces = os.environ.get('CURSUS_DEV_WORKSPACES', '').split(':')
        for workspace in dev_workspaces:
            if workspace and Path(workspace).exists():
                workspace_path = Path(workspace)
                if workspace_path not in workspace_dirs:
                    workspace_dirs.append(workspace_path)
        
        # Initialize with unified workspace list or package-only
        try:
            return StepCatalog(workspace_dirs=workspace_dirs if workspace_dirs else None)
        except Exception:
            # Silently ignore errors for optional enhancement
            return None

    def _detect_framework_if_needed(self, script_spec: ScriptExecutionSpec) -> Optional[str]:
        """Simple framework detection using step catalog (optional enhancement)."""
        if self.step_catalog:
            try:
                return self.step_catalog.detect_framework(script_spec.step_name)
            except Exception:
                # Silently ignore errors, return None for optional enhancement
                pass
        return None

    def _validate_builder_consistency_if_available(self, script_spec: ScriptExecutionSpec) -> List[str]:
        """Simple builder consistency check using step catalog (optional enhancement)."""
        warnings = []
        if self.step_catalog:
            try:
                builder_class = self.step_catalog.load_builder_class(script_spec.step_name)
                if builder_class and hasattr(builder_class, 'get_expected_input_paths'):
                    expected_inputs = builder_class.get_expected_input_paths()
                    script_inputs = set(script_spec.input_paths.keys())
                    missing_inputs = set(expected_inputs) - script_inputs
                    if missing_inputs:
                        warnings.append(f"Script missing expected input paths: {missing_inputs}")
            except Exception:
                # Silently ignore errors for optional enhancement
                pass
        return warnings

    def _discover_pipeline_components_if_needed(self, dag: 'PipelineDAG') -> Dict[str, Dict[str, Any]]:
        """Simple multi-workspace component discovery using step catalog (optional enhancement)."""
        if not self.step_catalog:
            return {}
            
        component_map = {}
        try:
            workspace_components = self.step_catalog.discover_cross_workspace_components()
            
            for node_name in dag.nodes:
                component_info = {
                    "node_name": node_name,
                    "available_workspaces": [],
                    "script_available": False,
                    "builder_available": False,
                    "contract_available": False
                }
                
                # Check each workspace for this component
                for workspace_id, components in workspace_components.items():
                    node_components = [c for c in components if node_name in c]
                    if node_components:
                        component_info["available_workspaces"].append(workspace_id)
                        for component in node_components:
                            if ":script" in component:
                                component_info["script_available"] = True
                            elif ":builder" in component:
                                component_info["builder_available"] = True
                            elif ":contract" in component:
                                component_info["contract_available"] = True
                
                component_map[node_name] = component_info
        except Exception:
            # Silently ignore errors for optional enhancement
            pass
        
        return component_map

    # Phase 2: Enhanced Testing Methods (User Stories Implementation)

    def test_script_with_step_catalog_enhancements(self, script_spec: ScriptExecutionSpec, main_params: Dict[str, Any]) -> ScriptTestResult:
        """
        US1: Enhanced script testing with optional step catalog features.
        
        Provides framework detection and builder consistency validation when step catalog available,
        falls back to standard testing when not available.
        """
        # Standard script testing (unchanged)
        result = self.test_script_with_spec(script_spec, main_params)
        
        # Optional step catalog enhancements
        if self.step_catalog and result.success:
            # Simple framework detection (internal use only)
            framework = self._detect_framework_if_needed(script_spec)
            
            # Simple builder consistency check (internal use only)
            consistency_warnings = self._validate_builder_consistency_if_available(script_spec)
            
            # Note: Step catalog enhancements are performed internally but don't modify
            # the ScriptTestResult structure to maintain API compatibility
        
        return result

    def test_data_compatibility_with_step_catalog_enhancements(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec) -> DataCompatibilityResult:
        """
        US2: Enhanced compatibility testing with optional contract awareness.
        
        Uses contract information for enhanced compatibility testing when step catalog available,
        falls back to standard semantic matching when not available.
        """
        # Try contract-aware compatibility first if step catalog available
        if self.step_catalog:
            try:
                contract_a = self.step_catalog.load_contract_class(spec_a.step_name)
                contract_b = self.step_catalog.load_contract_class(spec_b.step_name)
                
                if contract_a and contract_b:
                    # Use contract information for enhanced compatibility testing
                    return self._test_contract_aware_compatibility(spec_a, spec_b, contract_a, contract_b)
            except Exception:
                # Silently ignore errors and fall back to standard testing
                pass
        
        # Fallback to standard semantic matching
        return self.test_data_compatibility_with_specs(spec_a, spec_b)

    def test_pipeline_flow_with_step_catalog_enhancements(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
        """
        US3: Enhanced pipeline testing with optional multi-workspace support.
        
        Adds workspace analysis and framework detection when step catalog available,
        uses standard pipeline testing as base functionality.
        """
        # Standard pipeline testing (unchanged)
        results = self.test_pipeline_flow_with_spec(pipeline_spec)
        
        # Optional step catalog enhancements
        if self.step_catalog:
            try:
                # Simple multi-workspace component discovery
                component_analysis = self._discover_pipeline_components_if_needed(pipeline_spec.dag)
                if component_analysis:
                    # Framework analysis for each component
                    framework_analysis = {}
                    for node_name in pipeline_spec.dag.nodes:
                        if node_name in pipeline_spec.script_specs:
                            script_spec = pipeline_spec.script_specs[node_name]
                            framework = self._detect_framework_if_needed(script_spec)
                            if framework:
                                framework_analysis[node_name] = framework
                    
                    results["step_catalog_analysis"] = {
                        "workspace_analysis": component_analysis,
                        "framework_analysis": framework_analysis
                    }
            except Exception:
                # Silently ignore errors for optional enhancement
                pass
        
        return results

    def _test_contract_aware_compatibility(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec, contract_a: Any, contract_b: Any) -> DataCompatibilityResult:
        """Test compatibility using contract specifications."""
        try:
            # Execute script A
            main_params_a = self.builder.get_script_main_params(spec_a)
            script_a_result = self.test_script_with_spec(spec_a, main_params_a)
            
            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
                )
            
            # Get contract output specifications
            output_specs = {}
            if hasattr(contract_a, 'get_output_specifications'):
                output_specs = contract_a.get_output_specifications()
            elif hasattr(contract_a, 'get_output_paths'):
                # Fallback to path-based specifications
                output_paths = contract_a.get_output_paths()
                if output_paths:
                    output_specs = {name: {"type": "data"} for name in output_paths.keys()}
            
            # Get contract input specifications
            input_specs = {}
            if hasattr(contract_b, 'get_input_specifications'):
                input_specs = contract_b.get_input_specifications()
            elif hasattr(contract_b, 'get_input_paths'):
                # Fallback to path-based specifications
                input_paths = contract_b.get_input_paths()
                if input_paths:
                    input_specs = {name: {"type": "data"} for name in input_paths.keys()}
            
            # Match outputs to inputs using contract specifications
            compatibility_issues = []
            
            if output_specs and input_specs:
                for output_name, output_spec in output_specs.items():
                    for input_name, input_spec in input_specs.items():
                        if self._are_contract_specs_compatible(output_spec, input_spec):
                            # Found compatible pair, test actual data flow
                            return self._test_contract_data_flow(spec_a, spec_b, output_name, input_name)
                
                # No compatible contract specifications found
                compatibility_issues = [
                    "No compatible contract specifications found between outputs and inputs",
                    f"Available outputs: {list(output_specs.keys())}",
                    f"Available inputs: {list(input_specs.keys())}"
                ]
            else:
                compatibility_issues = ["Contract specifications not available for compatibility testing"]
            
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=compatibility_issues
            )
            
        except Exception as e:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Contract-aware compatibility test failed: {str(e)}"]
            )

    def _are_contract_specs_compatible(self, output_spec: Dict[str, Any], input_spec: Dict[str, Any]) -> bool:
        """Check if contract output and input specifications are compatible."""
        # Simple compatibility check based on type
        output_type = output_spec.get("type", "unknown")
        input_type = input_spec.get("type", "unknown")
        
        # Basic type compatibility
        if output_type == input_type:
            return True
        
        # Common compatible types
        compatible_pairs = [
            ("data", "dataset"),
            ("model", "artifact"),
            ("metrics", "evaluation"),
            ("processed_data", "data"),
            ("training_data", "data")
        ]
        
        for out_type, in_type in compatible_pairs:
            if (output_type == out_type and input_type == in_type) or \
               (output_type == in_type and input_type == out_type):
                return True
        
        return False

    def _test_contract_data_flow(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec, output_name: str, input_name: str) -> DataCompatibilityResult:
        """Test actual data flow between scripts using contract information."""
        try:
            # Get actual output directory from spec_a
            output_dir_a = Path(spec_a.output_paths.get(output_name, spec_a.output_paths.get("data_output", "")))
            output_files = self._find_valid_output_files(output_dir_a)
            
            if not output_files:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[f"No valid output files found for {output_name}"]
                )
            
            # Create modified spec_b with contract-aware input path
            modified_input_paths = spec_b.input_paths.copy()
            modified_input_paths[input_name] = str(output_files[0])  # Use first valid output file
            
            modified_spec_b = ScriptExecutionSpec(
                script_name=spec_b.script_name,
                step_name=spec_b.step_name,
                script_path=spec_b.script_path,
                input_paths=modified_input_paths,
                output_paths=spec_b.output_paths,
                environ_vars=spec_b.environ_vars,
                job_args=spec_b.job_args,
            )
            
            # Test script B with contract-aware input
            main_params_b = self.builder.get_script_main_params(modified_spec_b)
            script_b_result = self.test_script_with_spec(modified_spec_b, main_params_b)
            
            if script_b_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=True,
                    compatibility_issues=[],
                    data_format_a=self._detect_file_format(output_files[0]),
                    data_format_b=self._detect_file_format(output_files[0]),
                )
            else:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[f"Contract data flow test failed: {script_b_result.error_message}"]
                )
                
        except Exception as e:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Contract data flow test error: {str(e)}"]
            )

    def test_script_with_spec(
        self, script_spec: ScriptExecutionSpec, main_params: Dict[str, Any]
    ) -> ScriptTestResult:
        """Test script functionality using ScriptExecutionSpec"""
        start_time = time.time()

        try:
            script_path = self._find_script_path(script_spec.script_name)

            # Import script using standard Python import
            spec = importlib.util.spec_from_file_location("script", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for main function with correct signature
            has_main = hasattr(module, "main") and callable(module.main)

            if not has_main:
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="Script missing main() function",
                    execution_time=time.time() - start_time,
                    has_main_function=False,
                )

            # Validate main function signature matches script development guide
            sig = inspect.signature(module.main)
            expected_params = [
                "input_paths",
                "output_paths",
                "environ_vars",
                "job_args",
            ]
            actual_params = list(sig.parameters.keys())

            if not all(param in actual_params for param in expected_params):
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="Main function signature doesn't match script development guide",
                    execution_time=time.time() - start_time,
                    has_main_function=True,
                )

            # Create test directories based on ScriptExecutionSpec - use first available output path
            first_output_path = next(iter(script_spec.output_paths.values()))
            test_dir = Path(first_output_path)
            test_dir.mkdir(parents=True, exist_ok=True)

            # Validate that all required input paths exist - NO SAMPLE DATA GENERATION
            missing_inputs = []
            for logical_name, input_path in script_spec.input_paths.items():
                if not Path(input_path).exists():
                    missing_inputs.append(f"{logical_name}: {input_path}")

            if missing_inputs:
                error_details = [
                    f"Script '{script_spec.script_name}' requires the following input data:",
                    *[f"  - {item}" for item in missing_inputs],
                    "",
                    "Please ensure all required input data files exist before running the test.",
                    "You can check the ScriptExecutionSpec to see what input paths are expected."
                ]
                
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="\n".join(error_details),
                    execution_time=time.time() - start_time,
                    has_main_function=True,
                )

            # EXECUTE THE MAIN FUNCTION with ScriptExecutionSpec parameters
            module.main(**main_params)

            return ScriptTestResult(
                script_name=script_spec.script_name,
                success=True,
                error_message=None,
                execution_time=time.time() - start_time,
                has_main_function=True,
            )

        except Exception as e:
            return ScriptTestResult(
                script_name=script_spec.script_name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                has_main_function=has_main if "has_main" in locals() else False,
            )

    def test_data_compatibility_with_specs(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ) -> DataCompatibilityResult:
        """
        Enhanced data compatibility testing with intelligent path matching

        Uses logical name matching when available, falls back to semantic matching.
        This provides the best of both worlds - sophisticated matching when possible,
        with graceful degradation for backward compatibility.
        """
        
        # Use logical name matching if available (preferred approach)
        if self.enable_logical_matching:
            return self._test_data_compatibility_with_logical_matching(spec_a, spec_b)
        
        # Fall back to semantic matching for backward compatibility
        return self._test_data_compatibility_with_semantic_matching(spec_a, spec_b)

    def _test_data_compatibility_with_semantic_matching(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ) -> DataCompatibilityResult:
        """
        Test data compatibility using semantic path matching between output and input paths.
        
        This method uses the SemanticMatcher to intelligently connect spec_a's output paths
        to spec_b's input paths, eliminating hardcoded assumptions about logical names.
        """
        try:
            # Execute script A using its ScriptExecutionSpec
            main_params_a = self.builder.get_script_main_params(spec_a)
            script_a_result = self.test_script_with_spec(spec_a, main_params_a)

            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[
                        f"Script A failed: {script_a_result.error_message}"
                    ],
                )

            # Find semantic matches between A's outputs and B's inputs
            path_matches = self._find_semantic_path_matches(spec_a, spec_b)

            if not path_matches:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[
                        "No semantic matches found between output and input paths",
                        f"Available outputs from {spec_a.script_name}: {list(spec_a.output_paths.keys())}",
                        f"Available inputs for {spec_b.script_name}: {list(spec_b.input_paths.keys())}"
                    ],
                )

            # Try each match until we find one that works
            compatibility_issues = []
            
            for output_name, input_name, score in path_matches:
                try:
                    # Get actual output directory from spec_a
                    output_dir_a = Path(spec_a.output_paths[output_name])
                    output_files = self._find_valid_output_files(output_dir_a)

                    if not output_files:
                        compatibility_issues.append(
                            f"No valid output files found in {output_name} ({output_dir_a})"
                        )
                        continue  # Try next match

                    # Create modified spec_b with matched input path
                    modified_input_paths = spec_b.input_paths.copy()
                    modified_input_paths[input_name] = str(output_files[0])  # Use first valid output file

                    modified_spec_b = ScriptExecutionSpec(
                        script_name=spec_b.script_name,
                        step_name=spec_b.step_name,
                        script_path=spec_b.script_path,
                        input_paths=modified_input_paths,
                        output_paths=spec_b.output_paths,
                        environ_vars=spec_b.environ_vars,
                        job_args=spec_b.job_args,
                    )

                    # Test script B with matched input
                    main_params_b = self.builder.get_script_main_params(modified_spec_b)
                    script_b_result = self.test_script_with_spec(modified_spec_b, main_params_b)

                    if script_b_result.success:
                        return DataCompatibilityResult(
                            script_a=spec_a.script_name,
                            script_b=spec_b.script_name,
                            compatible=True,
                            compatibility_issues=[],
                            data_format_a=self._detect_file_format(output_files[0]),
                            data_format_b=self._detect_file_format(output_files[0]),
                        )
                    else:
                        compatibility_issues.append(
                            f"Match {output_name} -> {input_name} (score: {score:.3f}) failed: {script_b_result.error_message}"
                        )

                except Exception as match_error:
                    compatibility_issues.append(
                        f"Error testing match {output_name} -> {input_name}: {str(match_error)}"
                    )
                    continue  # Try next match

            # If no matches worked
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[
                    f"No working path matches found. Tried {len(path_matches)} semantic matches."
                ] + compatibility_issues,
            )

        except Exception as e:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Semantic compatibility test failed: {str(e)}"],
            )

    def _find_semantic_path_matches(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ) -> List[tuple]:
        """
        Find semantic matches between spec_a's output_paths and spec_b's input_paths.
        
        Returns:
            List of (output_logical_name, input_logical_name, similarity_score) tuples
            sorted by similarity score (highest first)
        """
        try:
            from ...core.deps.semantic_matcher import SemanticMatcher
        except ImportError:
            # Fallback to simple string matching if SemanticMatcher is not available
            return self._find_simple_path_matches(spec_a, spec_b)
        
        matcher = SemanticMatcher()
        matches = []
        
        # Match each output of spec_a to each input of spec_b
        for output_name in spec_a.output_paths.keys():
            for input_name in spec_b.input_paths.keys():
                score = matcher.calculate_similarity(output_name, input_name)
                if score > 0.3:  # Minimum threshold for meaningful matches
                    matches.append((output_name, input_name, score))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def _find_simple_path_matches(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ) -> List[tuple]:
        """
        Fallback simple string matching when SemanticMatcher is not available.
        
        Returns:
            List of (output_logical_name, input_logical_name, similarity_score) tuples
        """
        from difflib import SequenceMatcher
        
        matches = []
        
        # Match each output of spec_a to each input of spec_b
        for output_name in spec_a.output_paths.keys():
            for input_name in spec_b.input_paths.keys():
                # Simple string similarity
                score = SequenceMatcher(None, output_name.lower(), input_name.lower()).ratio()
                
                # Boost score for common semantic patterns
                if "data" in output_name.lower() and "data" in input_name.lower():
                    score += 0.2
                if "model" in output_name.lower() and "model" in input_name.lower():
                    score += 0.2
                if "eval" in output_name.lower() and ("eval" in input_name.lower() or "data" in input_name.lower()):
                    score += 0.2
                
                # Cap score at 1.0
                score = min(score, 1.0)
                
                if score > 0.3:  # Minimum threshold
                    matches.append((output_name, input_name, score))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def _test_data_compatibility_with_logical_matching(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ) -> DataCompatibilityResult:
        """Enhanced data compatibility testing using the logical_name_matching system"""

        try:
            # Execute script A using its ScriptExecutionSpec
            main_params_a = self.builder.get_script_main_params(spec_a)
            script_a_result = self.test_script_with_spec(spec_a, main_params_a)

            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[
                        f"Script A failed: {script_a_result.error_message}"
                    ],
                )

            # Find output files from script A
            # Use first available output path as starting point
            first_output_name = next(iter(spec_a.output_paths.keys()))
            output_dir_a = Path(spec_a.output_paths[first_output_name])
            output_files = self._find_valid_output_files(output_dir_a)

            if not output_files:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[
                        "Script A did not produce any valid output files"
                    ],
                )

            # Convert to enhanced specs and delegate to logical name matching system
            enhanced_spec_a = self._convert_to_enhanced_spec(spec_a)
            enhanced_spec_b = self._convert_to_enhanced_spec(spec_b)

            # Use the sophisticated logical name matching system
            enhanced_result = self.logical_name_tester.test_data_compatibility_with_logical_matching(
                enhanced_spec_a, enhanced_spec_b, output_files
            )

            # Convert enhanced result back to standard DataCompatibilityResult for API consistency
            return DataCompatibilityResult(
                script_a=enhanced_result.script_a,
                script_b=enhanced_result.script_b,
                compatible=enhanced_result.compatible,
                compatibility_issues=enhanced_result.compatibility_issues,
                data_format_a=enhanced_result.data_format_a,
                data_format_b=enhanced_result.data_format_b,
            )

        except Exception as e:
            # Fall back to semantic matching if logical matching fails
            return self._test_data_compatibility_with_semantic_matching(spec_a, spec_b)

    def test_pipeline_flow_with_spec(
        self, pipeline_spec: PipelineTestingSpec
    ) -> Dict[str, Any]:
        """
        Phase 3: Enhanced pipeline flow testing with topological ordering and data flow chaining

        This method now uses topological execution order when logical matching is available,
        falling back to the original approach for backward compatibility.
        """

        # Use enhanced topological execution if available
        if self.enable_logical_matching:
            return self._test_pipeline_flow_with_topological_ordering(pipeline_spec)

        # Fallback to original implementation for backward compatibility
        return self._test_pipeline_flow_original(pipeline_spec)

    def _test_pipeline_flow_with_topological_ordering(
        self, pipeline_spec: PipelineTestingSpec
    ) -> Dict[str, Any]:
        """Enhanced pipeline flow testing with topological ordering and data flow chaining"""

        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "execution_order": [],
            "errors": [],
        }

        try:
            dag = pipeline_spec.dag
            script_specs = pipeline_spec.script_specs

            if not dag.nodes:
                results["pipeline_success"] = False
                results["errors"].append("No nodes found in pipeline DAG")
                return results

            # Get topological execution order
            try:
                execution_order = dag.topological_sort()
                results["execution_order"] = execution_order
            except ValueError as e:
                results["pipeline_success"] = False
                results["errors"].append(f"DAG topology error: {str(e)}")
                return results

            # Execute in topological order, testing each node and its outgoing edges
            executed_nodes = set()

            for current_node in execution_order:
                if current_node not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"No ScriptExecutionSpec found for node: {current_node}"
                    )
                    continue

                # Test individual script functionality first
                script_spec = script_specs[current_node]
                main_params = self.builder.get_script_main_params(script_spec)

                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][current_node] = script_result

                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"Script {current_node} failed: {script_result.error_message}"
                    )
                    continue  # Skip data flow testing for failed scripts

                executed_nodes.add(current_node)

                # Test data compatibility with all dependent nodes
                outgoing_edges = [
                    (src, dst) for src, dst in dag.edges if src == current_node
                ]

                for src_node, dst_node in outgoing_edges:
                    if dst_node not in script_specs:
                        results["pipeline_success"] = False
                        results["errors"].append(
                            f"Missing ScriptExecutionSpec for destination node: {dst_node}"
                        )
                        continue

                    spec_a = script_specs[src_node]
                    spec_b = script_specs[dst_node]

                    # Test data compatibility using enhanced matching
                    compat_result = self.test_data_compatibility_with_specs(
                        spec_a, spec_b
                    )
                    results["data_flow_results"][
                        f"{src_node}->{dst_node}"
                    ] = compat_result

                    if not compat_result.compatible:
                        results["pipeline_success"] = False
                        results["errors"].extend(compat_result.compatibility_issues)

            # Validate all edges were tested
            expected_edges = set(f"{src}->{dst}" for src, dst in dag.edges)
            tested_edges = set(results["data_flow_results"].keys())
            missing_edges = expected_edges - tested_edges

            if missing_edges:
                results["pipeline_success"] = False
                results["errors"].append(f"Untested edges: {', '.join(missing_edges)}")

            return results

        except Exception as e:
            results["pipeline_success"] = False
            results["errors"].append(f"Enhanced pipeline flow test failed: {str(e)}")
            return results

    def _test_pipeline_flow_original(
        self, pipeline_spec: PipelineTestingSpec
    ) -> Dict[str, Any]:
        """Original pipeline flow testing implementation (backward compatibility)"""

        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "errors": [],
        }

        try:
            dag = pipeline_spec.dag
            script_specs = pipeline_spec.script_specs

            if not dag.nodes:
                results["pipeline_success"] = False
                results["errors"].append("No nodes found in pipeline DAG")
                return results

            # Test each script individually first using ScriptExecutionSpec
            for node_name in dag.nodes:
                if node_name not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"No ScriptExecutionSpec found for node: {node_name}"
                    )
                    continue

                script_spec = script_specs[node_name]
                main_params = self.builder.get_script_main_params(script_spec)

                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][node_name] = script_result

                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"Script {node_name} failed: {script_result.error_message}"
                    )

            # Test data flow between connected scripts using DAG edges
            for edge in dag.edges:
                script_a, script_b = edge

                if script_a not in script_specs or script_b not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"Missing ScriptExecutionSpec for edge: {script_a} -> {script_b}"
                    )
                    continue

                spec_a = script_specs[script_a]
                spec_b = script_specs[script_b]

                # Test data compatibility using ScriptExecutionSpecs
                compat_result = self.test_data_compatibility_with_specs(spec_a, spec_b)
                results["data_flow_results"][f"{script_a}->{script_b}"] = compat_result

                if not compat_result.compatible:
                    results["pipeline_success"] = False
                    results["errors"].extend(compat_result.compatibility_issues)

            return results

        except Exception as e:
            results["pipeline_success"] = False
            results["errors"].append(f"Pipeline flow test failed: {str(e)}")
            return results

    def _find_script_path(self, script_name: str) -> str:
        """
        Script discovery with workspace_dir prioritization - ESSENTIAL UTILITY

        Priority order:
        1. workspace_dir/{script_name}.py
        2. workspace_dir/scripts/{script_name}.py
        3. Original fallback locations
        """
        # Priority 1 & 2: Local workspace searches
        workspace_paths = [
            self.workspace_dir / f"{script_name}.py",
            self.workspace_dir / "scripts" / f"{script_name}.py",
        ]

        for path in workspace_paths:
            if path.exists():
                return str(path)

        # Priority 3: Original fallback locations (for backward compatibility)
        fallback_paths = [
            f"src/cursus/steps/scripts/{script_name}.py",
            f"scripts/{script_name}.py",
            f"dockers/xgboost_atoz/scripts/{script_name}.py",
            f"dockers/pytorch_bsm_ext/scripts/{script_name}.py",
        ]

        for path in fallback_paths:
            if Path(path).exists():
                return path

        raise FileNotFoundError(
            f"Script not found: {script_name}. Searched in workspace_dir ({self.workspace_dir}) and fallback locations."
        )

    def _is_temp_or_system_file(self, file_path: Path) -> bool:
        """
        Check if a file is a temporary or system file that should be excluded from output detection.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be excluded, False otherwise
        """
        filename = file_path.name.lower()

        # Temporary file patterns
        temp_patterns = [
            r".*\.tmp$",  # .tmp files
            r".*\.temp$",  # .temp files
            r".*~$",  # backup files ending with ~
            r".*\.swp$",  # vim swap files
            r".*\.bak$",  # backup files
            r".*\.orig$",  # original files from merges
            r".*\.rej$",  # rejected patches
            r".*\.lock$",  # lock files
            r".*\.pid$",  # process ID files
            r".*\.log$",  # log files (usually not data outputs)
        ]

        # System files
        system_files = {
            ".ds_store",  # macOS
            "thumbs.db",  # Windows
            "desktop.ini",  # Windows
        }

        # Hidden files (starting with .) - but allow some exceptions
        if filename.startswith(".") and filename not in {".gitkeep", ".placeholder"}:
            return True

        # Check against system files
        if filename in system_files:
            return True

        # Check against temp patterns
        for pattern in temp_patterns:
            if re.match(pattern, filename):
                return True

        # Check for cache directories and files
        if "__pycache__" in str(file_path) or filename.endswith((".pyc", ".pyo")):
            return True

        return False

    def _find_valid_output_files(
        self, output_dir: Path, min_size_bytes: int = 1
    ) -> List[Path]:
        """
        Find valid output files in a directory, excluding temporary and system files.

        Args:
            output_dir: Directory to search for output files
            min_size_bytes: Minimum file size to consider (default 1 byte, excludes empty files)

        Returns:
            List of valid output file paths, sorted by modification time (newest first)
        """
        if not output_dir.exists() or not output_dir.is_dir():
            return []

        valid_files = []

        for file_path in output_dir.iterdir():
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip temporary/system files
            if self._is_temp_or_system_file(file_path):
                continue

            # Check file size
            try:
                if file_path.stat().st_size < min_size_bytes:
                    continue
            except (OSError, IOError):
                # Skip files we can't stat
                continue

            valid_files.append(file_path)

        # Sort by modification time, newest first
        valid_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        return valid_files

    # Phase 2: Inference Testing Methods (4 Core Functionalities)

    def test_inference_function(self, handler_module: Any, function_name: str, 
                               test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual inference function (model_fn, input_fn, predict_fn, output_fn)."""
        start_time = time.time()
        
        try:
            # Get function from module
            func = getattr(handler_module, function_name)
            
            # Execute function with test parameters
            result = func(**test_params)
            
            # Validate result based on function type
            validation = self._validate_function_result(function_name, result, test_params)
            
            return {
                "function_name": function_name,
                "success": True,
                "execution_time": time.time() - start_time,
                "result": result,
                "validation": validation
            }
        except Exception as e:
            return {
                "function_name": function_name,
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }

    def test_inference_pipeline(self, handler_spec: InferenceHandlerSpec) -> Dict[str, Any]:
        """Test complete inference pipeline (all 4 functions connected)."""
        results = {
            "pipeline_success": True, 
            "function_results": {}, 
            "errors": [],
            "handler_name": handler_spec.handler_name,
            "step_name": handler_spec.step_name
        }
        
        try:
            # Extract packaged model to inference_inputs/
            extraction_paths = self._extract_packaged_model(
                handler_spec.packaged_model_path, 
                "inference_inputs"
            )
            
            # Load inference handler from extracted code/
            handler_module = self._load_handler_module(
                extraction_paths["handler_file"]
            )
            
            # Load payload samples from payload_samples_path
            payload_samples = self._load_payload_samples(
                handler_spec.payload_samples_path
            )
            
            # Step 1: Test model_fn with extraction root (model files at root level)
            model_artifacts = handler_module.model_fn(extraction_paths["extraction_root"])
            results["function_results"]["model_fn"] = {"success": True, "artifacts": model_artifacts}
            
            # Initialize function result containers
            input_fn_results = []
            predict_fn_results = []
            output_fn_results = []
            end_to_end_results = []
            
            # Step 2-4: Test pipeline with payload samples
            for sample in payload_samples:
                try:
                    # Step 2: input_fn
                    processed_input = handler_module.input_fn(sample["data"], sample["content_type"])
                    input_fn_results.append({"success": True, "sample": sample["sample_name"]})
                    
                    # Step 3: predict_fn
                    predictions = handler_module.predict_fn(processed_input, model_artifacts)
                    predict_fn_results.append({"success": True, "sample": sample["sample_name"]})
                    
                    # Step 4: output_fn
                    for accept_type in handler_spec.supported_accept_types:
                        response = handler_module.output_fn(predictions, accept_type)
                        output_fn_results.append({
                            "success": True, 
                            "sample": sample["sample_name"],
                            "accept_type": accept_type
                        })
                    
                    # End-to-end success for this sample
                    end_to_end_results.append({"success": True, "sample": sample["sample_name"]})
                    
                except Exception as sample_error:
                    # Record individual function failures
                    input_fn_results.append({"success": False, "sample": sample["sample_name"], "error": str(sample_error)})
                    predict_fn_results.append({"success": False, "sample": sample["sample_name"], "error": str(sample_error)})
                    output_fn_results.append({"success": False, "sample": sample["sample_name"], "error": str(sample_error)})
                    end_to_end_results.append({"success": False, "sample": sample["sample_name"], "error": str(sample_error)})
                    
                    results["pipeline_success"] = False
                    results["errors"].append(f"Sample {sample['sample_name']} failed: {str(sample_error)}")
            
            # Store individual function results
            results["function_results"]["input_fn"] = input_fn_results
            results["function_results"]["predict_fn"] = predict_fn_results  
            results["function_results"]["output_fn"] = output_fn_results
            results["function_results"]["end_to_end"] = end_to_end_results
            
        except Exception as e:
            results["pipeline_success"] = False
            error_msg = f"Handler '{handler_spec.handler_name}' ({handler_spec.step_name}) failed: {str(e)}"
            results["errors"].append(error_msg)
        finally:
            # Cleanup extraction directory
            self._cleanup_extraction_directory("inference_inputs")
        
        return results

    def test_script_to_inference_compatibility(self, script_spec: ScriptExecutionSpec,
                                              handler_spec: InferenceHandlerSpec) -> Dict[str, Any]:
        """Test data compatibility between script output and inference input."""
        
        # Execute script first
        script_result = self.test_script_with_spec(script_spec, self.builder.get_script_main_params(script_spec))
        
        if not script_result.success:
            return {"compatible": False, "error": "Script execution failed"}
        
        # Find script output files using semantic matching (like existing RuntimeTester)
        # Use semantic matching to find the best output path for inference input
        output_files = []
        compatibility_issues = []
        
        # Try each output path from script_spec to find valid files
        for output_name, output_path in script_spec.output_paths.items():
            output_dir = Path(output_path)
            files = self._find_valid_output_files(output_dir)
            if files:
                output_files.extend(files)
                break  # Use first valid output path
            else:
                compatibility_issues.append(f"No valid files in output path '{output_name}': {output_path}")
        
        if not output_files:
            return {
                "compatible": False, 
                "error": "No script output files found",
                "details": compatibility_issues
            }
        
        # Test if inference handler can process script output
        try:
            # Extract packaged model and load handler
            extraction_paths = self._extract_packaged_model(
                handler_spec.packaged_model_path, 
                "inference_inputs"
            )
            handler_module = self._load_handler_module(extraction_paths["handler_file"])
            
            # Try each output file with different content types
            for output_file in output_files:
                try:
                    with open(output_file, 'r') as f:
                        script_output_data = f.read()
                    
                    # Test with different content types
                    for content_type in handler_spec.supported_content_types:
                        try:
                            processed_input = handler_module.input_fn(script_output_data, content_type)
                            return {
                                "compatible": True, 
                                "content_type": content_type,
                                "output_file": str(output_file),
                                "file_format": self._detect_file_format(output_file)
                            }
                        except Exception as content_error:
                            compatibility_issues.append(
                                f"Content type '{content_type}' failed for {output_file.name}: {str(content_error)}"
                            )
                            continue
                            
                except Exception as file_error:
                    compatibility_issues.append(f"Failed to read {output_file.name}: {str(file_error)}")
                    continue
            
            return {
                "compatible": False, 
                "error": "No compatible content type found for any output file",
                "details": compatibility_issues
            }
            
        except Exception as e:
            return {"compatible": False, "error": str(e)}
        finally:
            self._cleanup_extraction_directory("inference_inputs")

    def test_pipeline_with_inference(self, pipeline_spec: PipelineTestingSpec,
                                    inference_handlers: Dict[str, InferenceHandlerSpec]) -> Dict[str, Any]:
        """Test pipeline where inference handlers replace registration steps."""
        
        results = {"pipeline_success": True, "script_results": {}, "inference_results": {}, "errors": []}
        
        # Test scripts first
        for node_name, script_spec in pipeline_spec.script_specs.items():
            if node_name not in inference_handlers:  # Only test non-inference scripts
                main_params = self.builder.get_script_main_params(script_spec)
                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][node_name] = script_result
                
                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {node_name} failed")
        
        # Test inference handlers
        for node_name, handler_spec in inference_handlers.items():
            handler_result = self.test_inference_pipeline(handler_spec)
            results["inference_results"][node_name] = handler_result
            
            if not handler_result["pipeline_success"]:
                results["pipeline_success"] = False
                results["errors"].extend(handler_result["errors"])
        
        # Test data flow between scripts and inference handlers
        for src_node, dst_node in pipeline_spec.dag.edges:
            if src_node in pipeline_spec.script_specs and dst_node in inference_handlers:
                compatibility = self.test_script_to_inference_compatibility(
                    pipeline_spec.script_specs[src_node],
                    inference_handlers[dst_node]
                )
                if not compatibility["compatible"]:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Incompatible data flow: {src_node} -> {dst_node}")
        
        return results

    # Helper methods for inference testing
    def _extract_packaged_model(self, packaged_model_path: str, extraction_dir: str = "inference_inputs") -> Dict[str, str]:
        """Extract model.tar.gz and return paths to key components."""
        import tarfile
        
        extraction_path = Path(extraction_dir)
        extraction_path.mkdir(parents=True, exist_ok=True)
        
        # Extract tar.gz to extraction directory
        with tarfile.open(packaged_model_path, "r:gz") as tar:
            tar.extractall(path=extraction_path)
        
        # Return key paths based on package step structure
        paths = {
            "extraction_root": str(extraction_path),
            "inference_code": str(extraction_path / "code"),
            "handler_file": str(extraction_path / "code" / "inference.py")  # Assuming standard name
        }
        
        # Check for optional calibration
        calibration_dir = extraction_path / "calibration"
        if calibration_dir.exists():
            paths["calibration_model"] = str(calibration_dir)
        
        return paths

    def _load_handler_module(self, handler_file_path: str):
        """Load inference handler module (similar to existing _find_script_path pattern)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference_handler", handler_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_payload_samples(self, payload_samples_path: str) -> List[Dict[str, Any]]:
        """Load test samples from payload samples directory."""
        samples = []
        payload_dir = Path(payload_samples_path)
        
        # Load CSV samples
        csv_dir = payload_dir / "csv_samples"
        if csv_dir.exists():
            for csv_file in csv_dir.glob("*.csv"):
                with open(csv_file, 'r') as f:
                    samples.append({
                        "sample_name": csv_file.stem,
                        "content_type": "text/csv",
                        "data": f.read().strip(),
                        "file_path": str(csv_file)
                    })
        
        # Load JSON samples
        json_dir = payload_dir / "json_samples"
        if json_dir.exists():
            for json_file in json_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    samples.append({
                        "sample_name": json_file.stem,
                        "content_type": "application/json",
                        "data": f.read().strip(),
                        "file_path": str(json_file)
                    })
        
        return samples

    def _cleanup_extraction_directory(self, extraction_dir: str) -> None:
        """Clean up extraction directory after testing."""
        import shutil
        extraction_path = Path(extraction_dir)
        if extraction_path.exists():
            shutil.rmtree(extraction_path)

    def _validate_function_result(self, function_name: str, result: Any, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function result based on function type (reuses existing validation patterns)."""
        validation = {"function_type": function_name, "result_type": type(result).__name__}
        
        if function_name == "model_fn":
            validation["has_model_artifacts"] = result is not None
            validation["is_dict"] = isinstance(result, dict)
        elif function_name == "input_fn":
            validation["has_processed_input"] = result is not None
            validation["input_type"] = type(result).__name__
        elif function_name == "predict_fn":
            validation["has_predictions"] = result is not None
            validation["prediction_type"] = type(result).__name__
        elif function_name == "output_fn":
            validation["has_response"] = result is not None
            validation["response_type"] = type(result).__name__
        
        return validation

    # Phase 3: Helper Methods for Enhanced Functionality

    def _create_modified_spec_with_matches(
        self, spec_b: ScriptExecutionSpec, path_matches: List, output_files: List[Path]
    ) -> ScriptExecutionSpec:
        """Create modified spec_b with actual output file paths from script A based on logical name matches"""

        # Map logical names to actual file paths
        logical_to_file_map = {}
        for output_file in output_files:
            # Use file naming convention or metadata to map to logical names
            # This could be enhanced with more sophisticated mapping logic
            logical_to_file_map[output_file.stem] = str(output_file)

        # Create new input paths based on matches
        new_input_paths = spec_b.input_paths.copy()

        for match in path_matches:
            if hasattr(match, "source_logical_name") and hasattr(
                match, "dest_logical_name"
            ):
                source_name = match.source_logical_name
                dest_name = match.dest_logical_name

                # Find matching output file
                for output_file in output_files:
                    if (
                        source_name in output_file.stem
                        or output_file.stem in source_name
                    ):
                        new_input_paths[dest_name] = str(output_file)
                        break
                else:
                    # If no specific match found, use the first available output file
                    if output_files:
                        new_input_paths[dest_name] = str(output_files[0])

        # Return modified spec with updated input paths
        return ScriptExecutionSpec(
            script_name=spec_b.script_name,
            step_name=spec_b.step_name,
            script_path=spec_b.script_path,
            input_paths=new_input_paths,
            output_paths=spec_b.output_paths,
            environ_vars=spec_b.environ_vars,
            job_args=spec_b.job_args,
        )

    def _generate_matching_report(self, path_matches: List) -> Dict[str, Any]:
        """Generate detailed matching report for debugging and analysis"""

        if not path_matches:
            return {
                "total_matches": 0,
                "match_types": {},
                "confidence_distribution": {},
                "recommendations": [
                    "No logical name matches found. Consider standardizing naming conventions."
                ],
            }

        # Analyze match types
        match_types = {}
        confidence_scores = []

        for match in path_matches:
            if hasattr(match, "match_type") and hasattr(match, "confidence"):
                match_type = match.match_type
                confidence = match.confidence

                match_types[match_type] = match_types.get(match_type, 0) + 1
                confidence_scores.append(confidence)

        # Calculate confidence distribution
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            high_confidence = len([c for c in confidence_scores if c >= 0.8])
            medium_confidence = len([c for c in confidence_scores if 0.5 <= c < 0.8])
            low_confidence = len([c for c in confidence_scores if c < 0.5])
        else:
            avg_confidence = 0.0
            high_confidence = medium_confidence = low_confidence = 0

        # Generate recommendations
        recommendations = []
        if avg_confidence < 0.7:
            recommendations.append(
                "Average confidence is low. Consider adding more specific aliases."
            )
        if low_confidence > 0:
            recommendations.append(
                f"{low_confidence} matches have low confidence. Review naming conventions."
            )
        if len(match_types.get("semantic", 0)) > len(
            match_types.get("exact_logical", 0)
        ):
            recommendations.append(
                "Many semantic matches found. Consider standardizing logical names."
            )

        return {
            "total_matches": len(path_matches),
            "match_types": match_types,
            "confidence_distribution": {
                "average": avg_confidence,
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence,
            },
            "recommendations": (
                recommendations if recommendations else ["Matching looks good!"]
            ),
        }

    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from file extension"""
        if not file_path or not isinstance(file_path, Path):
            return "unknown"

        suffix = file_path.suffix.lower()

        # Map common extensions to format names
        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".parquet": "parquet",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".bst": "xgboost_model",
            ".onnx": "onnx_model",
            ".tar.gz": "compressed_archive",
            ".zip": "compressed_archive",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".txt": "text",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
        }

        return format_map.get(suffix, suffix[1:] if suffix else "no_extension")

    # Phase 2: Logical Name Matching Methods (available when logical_name_matching module is present)

    def test_data_compatibility_with_logical_matching(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ):
        """
        Enhanced data compatibility testing with logical name matching

        Args:
            spec_a: Source script specification
            spec_b: Destination script specification

        Returns:
            EnhancedDataCompatibilityResult if logical matching is enabled, otherwise DataCompatibilityResult
        """
        if not self.enable_logical_matching:
            return self.test_data_compatibility_with_specs(spec_a, spec_b)

        # Execute script A first
        main_params_a = self.builder.get_script_main_params(spec_a)
        script_a_result = self.test_script_with_spec(spec_a, main_params_a)

        if not script_a_result.success:
            return EnhancedDataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[
                    f"Script A failed: {script_a_result.error_message}"
                ],
            )

        # Find valid output files using semantic matching to get the best output path
        path_matches = self._find_semantic_path_matches(spec_a, spec_b)
        if path_matches:
            # Use the best matching output path
            best_output_name = path_matches[0][0]  # First match has highest score
            output_dir_a = Path(spec_a.output_paths[best_output_name])
        else:
            # Fallback to first available output path
            first_output_name = next(iter(spec_a.output_paths.keys()))
            output_dir_a = Path(spec_a.output_paths[first_output_name])
        
        output_files = self._find_valid_output_files(output_dir_a)

        if not output_files:
            return EnhancedDataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[
                    "Script A did not produce any valid output files"
                ],
            )

        # Convert to enhanced specs and use logical name matching
        enhanced_spec_a = self._convert_to_enhanced_spec(spec_a)
        enhanced_spec_b = self._convert_to_enhanced_spec(spec_b)

        return self.logical_name_tester.test_data_compatibility_with_logical_matching(
            enhanced_spec_a, enhanced_spec_b, output_files
        )

    def test_pipeline_flow_with_topological_execution(
        self, pipeline_spec: PipelineTestingSpec
    ) -> Dict[str, Any]:
        """
        Enhanced pipeline flow testing with topological execution order

        Args:
            pipeline_spec: Pipeline testing specification

        Returns:
            Dictionary with comprehensive pipeline test results including execution order
        """
        if not self.enable_logical_matching:
            return self.test_pipeline_flow_with_spec(pipeline_spec)

        # Convert script specs to enhanced specs
        enhanced_script_specs = {}
        for node_name, script_spec in pipeline_spec.script_specs.items():
            enhanced_script_specs[node_name] = self._convert_to_enhanced_spec(
                script_spec
            )

        # Create script tester function
        def script_tester_func(
            enhanced_spec: EnhancedScriptExecutionSpec,
        ) -> ScriptTestResult:
            original_spec = self._convert_from_enhanced_spec(enhanced_spec)
            main_params = self.builder.get_script_main_params(original_spec)
            return self.test_script_with_spec(original_spec, main_params)

        return self.logical_name_tester.test_pipeline_with_topological_execution(
            pipeline_spec.dag, enhanced_script_specs, script_tester_func
        )

    def get_path_matches(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ):
        """
        Get logical name matches between two script specifications

        Args:
            spec_a: Source script specification
            spec_b: Destination script specification

        Returns:
            List of PathMatch objects if logical matching is enabled, empty list otherwise
        """
        if not self.enable_logical_matching:
            return []

        enhanced_spec_a = self._convert_to_enhanced_spec(spec_a)
        enhanced_spec_b = self._convert_to_enhanced_spec(spec_b)

        return self.path_matcher.find_path_matches(enhanced_spec_a, enhanced_spec_b)

    def generate_matching_report(
        self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec
    ) -> Dict[str, Any]:
        """
        Generate detailed matching report between two script specifications

        Args:
            spec_a: Source script specification
            spec_b: Destination script specification

        Returns:
            Dictionary with detailed matching information
        """
        if not self.enable_logical_matching:
            return {"error": "Logical name matching is not available"}

        path_matches = self.get_path_matches(spec_a, spec_b)
        return self.path_matcher.generate_matching_report(path_matches)

    def validate_pipeline_logical_names(
        self, pipeline_spec: PipelineTestingSpec
    ) -> Dict[str, Any]:
        """
        Validate logical name compatibility across entire pipeline

        Args:
            pipeline_spec: Pipeline testing specification

        Returns:
            Dictionary with validation results for all edges
        """
        if not self.enable_logical_matching:
            return {"error": "Logical name matching is not available"}

        validation_results = {
            "overall_valid": True,
            "edge_validations": {},
            "recommendations": [],
            "summary": {},
        }

        total_edges = 0
        valid_edges = 0

        for src_node, dst_node in pipeline_spec.dag.edges:
            total_edges += 1
            edge_key = f"{src_node}->{dst_node}"

            if (
                src_node not in pipeline_spec.script_specs
                or dst_node not in pipeline_spec.script_specs
            ):
                validation_results["edge_validations"][edge_key] = {
                    "valid": False,
                    "error": "Missing script specification",
                }
                validation_results["overall_valid"] = False
                continue

            spec_a = pipeline_spec.script_specs[src_node]
            spec_b = pipeline_spec.script_specs[dst_node]

            path_matches = self.get_path_matches(spec_a, spec_b)
            matching_report = self.generate_matching_report(spec_a, spec_b)

            edge_valid = len(path_matches) > 0
            if edge_valid:
                valid_edges += 1
            else:
                validation_results["overall_valid"] = False

            validation_results["edge_validations"][edge_key] = {
                "valid": edge_valid,
                "matches_found": len(path_matches),
                "high_confidence_matches": len(
                    [m for m in path_matches if m.confidence >= 0.8]
                ),
                "matching_report": matching_report,
            }

        validation_results["summary"] = {
            "total_edges": total_edges,
            "valid_edges": valid_edges,
            "validation_rate": valid_edges / total_edges if total_edges > 0 else 0.0,
        }

        if validation_results["summary"]["validation_rate"] < 1.0:
            validation_results["recommendations"].append(
                "Some edges have no logical name matches. Consider adding aliases or standardizing naming conventions."
            )

        return validation_results

    def _convert_to_enhanced_spec(
        self,
        original_spec: ScriptExecutionSpec,
        input_aliases: Optional[Dict[str, List[str]]] = None,
        output_aliases: Optional[Dict[str, List[str]]] = None,
    ):
        """Convert original ScriptExecutionSpec to EnhancedScriptExecutionSpec"""
        if not self.enable_logical_matching:
            return original_spec

        if input_aliases is None:
            input_aliases = self._generate_default_input_aliases(original_spec)
        if output_aliases is None:
            output_aliases = self._generate_default_output_aliases(original_spec)

        return EnhancedScriptExecutionSpec.from_script_execution_spec(
            original_spec, input_aliases, output_aliases
        )

    def _convert_from_enhanced_spec(self, enhanced_spec) -> ScriptExecutionSpec:
        """Convert EnhancedScriptExecutionSpec back to original ScriptExecutionSpec"""
        return ScriptExecutionSpec(
            script_name=enhanced_spec.script_name,
            step_name=enhanced_spec.step_name,
            script_path=enhanced_spec.script_path,
            input_paths=enhanced_spec.input_paths,
            output_paths=enhanced_spec.output_paths,
            environ_vars=enhanced_spec.environ_vars,
            job_args=enhanced_spec.job_args,
            last_updated=getattr(enhanced_spec, "last_updated", None),
            user_notes=getattr(enhanced_spec, "user_notes", None),
        )

    def _generate_default_input_aliases(
        self, spec: ScriptExecutionSpec
    ) -> Dict[str, List[str]]:
        """Generate default input aliases based on common patterns"""
        aliases = {}

        for logical_name in spec.input_paths.keys():
            alias_list = []

            # Common input aliases
            if "data" in logical_name.lower():
                alias_list.extend(
                    ["dataset", "input", "training_data", "processed_data"]
                )
            if "model" in logical_name.lower():
                alias_list.extend(["artifact", "trained_model", "model_input"])
            if "config" in logical_name.lower():
                alias_list.extend(
                    ["configuration", "params", "hyperparameters", "settings"]
                )

            # Add variations of the logical name
            if "_" in logical_name:
                alias_list.append(logical_name.replace("_", "-"))
                alias_list.append(logical_name.replace("_", ""))

            aliases[logical_name] = list(set(alias_list))  # Remove duplicates

        return aliases

    def _generate_default_output_aliases(
        self, spec: ScriptExecutionSpec
    ) -> Dict[str, List[str]]:
        """Generate default output aliases based on common patterns"""
        aliases = {}

        for logical_name in spec.output_paths.keys():
            alias_list = []

            # Common output aliases
            if "data" in logical_name.lower():
                alias_list.extend(["dataset", "output", "processed_data", "result"])
            if "model" in logical_name.lower():
                alias_list.extend(["artifact", "trained_model", "model_output"])
            if "evaluation" in logical_name.lower():
                alias_list.extend(["eval", "metrics", "results", "assessment"])

            # Add variations of the logical name
            if "_" in logical_name:
                alias_list.append(logical_name.replace("_", "-"))
                alias_list.append(logical_name.replace("_", ""))

            aliases[logical_name] = list(set(alias_list))  # Remove duplicates

        return aliases
