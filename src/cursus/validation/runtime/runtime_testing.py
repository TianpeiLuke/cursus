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
    RuntimeTestingConfiguration
)
from .runtime_spec_builder import PipelineTestingSpecBuilder

# Import PipelineDAG for integration
from ...api.dag.base_dag import PipelineDAG

# Import Phase 2 logical name matching components (optional)
try:
    from .logical_name_matching import (
        PathMatcher,
        TopologicalExecutor,
        LogicalNameMatchingTester,
        EnhancedScriptExecutionSpec,
        EnhancedDataCompatibilityResult
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
    
    def __init__(self, config_or_workspace_dir, enable_logical_matching: bool = True, semantic_threshold: float = 0.7):
        # Support both new RuntimeTestingConfiguration and old string workspace_dir for backward compatibility
        if isinstance(config_or_workspace_dir, RuntimeTestingConfiguration):
            self.config = config_or_workspace_dir
            self.pipeline_spec = config_or_workspace_dir.pipeline_spec
            self.workspace_dir = Path(config_or_workspace_dir.pipeline_spec.test_workspace_root)
            
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
        self.enable_logical_matching = enable_logical_matching and LOGICAL_MATCHING_AVAILABLE
        if self.enable_logical_matching:
            self.path_matcher = PathMatcher(semantic_threshold)
            self.topological_executor = TopologicalExecutor()
            self.logical_name_tester = LogicalNameMatchingTester(semantic_threshold)
        else:
            self.path_matcher = None
            self.topological_executor = None
            self.logical_name_tester = None
    
    def test_script_with_spec(self, script_spec: ScriptExecutionSpec, main_params: Dict[str, Any]) -> ScriptTestResult:
        """Test script functionality using ScriptExecutionSpec"""
        start_time = time.time()
        
        try:
            script_path = self._find_script_path(script_spec.script_name)
            
            # Import script using standard Python import
            spec = importlib.util.spec_from_file_location("script", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for main function with correct signature
            has_main = hasattr(module, 'main') and callable(module.main)
            
            if not has_main:
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="Script missing main() function",
                    execution_time=time.time() - start_time,
                    has_main_function=False
                )
            
            # Validate main function signature matches script development guide
            sig = inspect.signature(module.main)
            expected_params = ['input_paths', 'output_paths', 'environ_vars', 'job_args']
            actual_params = list(sig.parameters.keys())
            
            if not all(param in actual_params for param in expected_params):
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="Main function signature doesn't match script development guide",
                    execution_time=time.time() - start_time,
                    has_main_function=True
                )
            
            # Create test directories based on ScriptExecutionSpec
            test_dir = Path(script_spec.output_paths["data_output"])
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Use ScriptExecutionSpec input data path or generate sample data
            input_data_path = script_spec.input_paths.get("data_input")
            if not input_data_path or not Path(input_data_path).exists():
                # Generate sample data for testing
                sample_data = self._generate_sample_data()
                input_data_path = test_dir / "input_data.csv"
                pd.DataFrame(sample_data).to_csv(input_data_path, index=False)
            
            # EXECUTE THE MAIN FUNCTION with ScriptExecutionSpec parameters
            module.main(**main_params)
            
            return ScriptTestResult(
                script_name=script_spec.script_name,
                success=True,
                error_message=None,
                execution_time=time.time() - start_time,
                has_main_function=True
            )
            
        except Exception as e:
            return ScriptTestResult(
                script_name=script_spec.script_name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                has_main_function=has_main if 'has_main' in locals() else False
            )
    
    def test_data_compatibility_with_specs(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec) -> DataCompatibilityResult:
        """
        Phase 3: Enhanced data compatibility testing with intelligent path matching
        
        This method now uses logical name matching when available, falling back to 
        the original file-based approach for backward compatibility.
        """
        
        # Use enhanced logical name matching if available
        if self.enable_logical_matching:
            return self._test_data_compatibility_with_logical_matching(spec_a, spec_b)
        
        # Fallback to original implementation for backward compatibility
        return self._test_data_compatibility_original(spec_a, spec_b)
    
    def _test_data_compatibility_with_logical_matching(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec) -> DataCompatibilityResult:
        """Enhanced data compatibility testing with logical name matching"""
        
        try:
            # Execute script A using its ScriptExecutionSpec
            main_params_a = self.builder.get_script_main_params(spec_a)
            script_a_result = self.test_script_with_spec(spec_a, main_params_a)
            
            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
                )
            
            # Find valid output files from script A (any format, excluding temp files)
            output_dir_a = Path(spec_a.output_paths["data_output"])
            output_files = self._find_valid_output_files(output_dir_a)
            
            if not output_files:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=["Script A did not produce any valid output files"]
                )
            
            # Convert to enhanced specs for logical name matching
            enhanced_spec_a = self._convert_to_enhanced_spec(spec_a)
            enhanced_spec_b = self._convert_to_enhanced_spec(spec_b)
            
            # Find logical name matches using PathMatcher
            path_matches = self.path_matcher.find_path_matches(enhanced_spec_a, enhanced_spec_b)
            
            if not path_matches:
                # No logical matches found, fall back to original file-based approach
                return self._test_data_compatibility_original(spec_a, spec_b)
            
            # Create modified spec_b with matched paths
            modified_spec_b = self._create_modified_spec_with_matches(
                spec_b, path_matches, output_files
            )
            
            # Execute script B with matched inputs
            main_params_b = self.builder.get_script_main_params(modified_spec_b)
            script_b_result = self.test_script_with_spec(modified_spec_b, main_params_b)
            
            # Generate matching report
            matching_report = self._generate_matching_report(path_matches)
            
            # Return enhanced results with matching information
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=script_b_result.success,
                compatibility_issues=[] if script_b_result.success else [script_b_result.error_message],
                data_format_a=self._detect_file_format(output_files[0]) if output_files else 'unknown',
                data_format_b=self._detect_file_format(output_files[0]) if script_b_result.success and output_files else 'unknown'
            )
            
        except Exception as e:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Enhanced compatibility test failed: {str(e)}"]
            )
    
    def _test_data_compatibility_original(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec) -> DataCompatibilityResult:
        """Original data compatibility testing implementation (backward compatibility)"""
        
        try:
            # Execute script A using its ScriptExecutionSpec
            main_params_a = self.builder.get_script_main_params(spec_a)
            script_a_result = self.test_script_with_spec(spec_a, main_params_a)
            
            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
                )
            
            # Check if script A produced valid output files (any format, excluding temp files)
            output_dir_a = Path(spec_a.output_paths["data_output"])
            output_files = self._find_valid_output_files(output_dir_a)
            
            if not output_files:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=["Script A did not produce any valid output files"]
                )
            
            # Try each output file from script A as input to script B
            # Start with the most recently modified file (first in sorted list)
            compatibility_issues = []
            successful_tests = []
            
            for output_file in output_files:
                try:
                    # Create a modified spec_b with script A's output as input
                    modified_spec_b = ScriptExecutionSpec(
                        script_name=spec_b.script_name,
                        step_name=spec_b.step_name,
                        script_path=spec_b.script_path,
                        input_paths={"data_input": str(output_file)},  # Use script A's output
                        output_paths=spec_b.output_paths,
                        environ_vars=spec_b.environ_vars,
                        job_args=spec_b.job_args
                    )
                    
                    # Test script B with script A's output
                    main_params_b = self.builder.get_script_main_params(modified_spec_b)
                    script_b_result = self.test_script_with_spec(modified_spec_b, main_params_b)
                    
                    if script_b_result.success:
                        # Success! Record the working combination
                        successful_tests.append({
                            'output_file': output_file.name,
                            'format': output_file.suffix.lower() or 'no_extension'
                        })
                        break  # Found a working combination, no need to test others
                    else:
                        # Record the failure for this file
                        compatibility_issues.append(
                            f"Script B failed with output file '{output_file.name}' "
                            f"({output_file.suffix or 'no extension'}): {script_b_result.error_message}"
                        )
                        
                except Exception as file_test_error:
                    compatibility_issues.append(
                        f"Error testing with output file '{output_file.name}': {str(file_test_error)}"
                    )
            
            # Determine overall compatibility
            is_compatible = len(successful_tests) > 0
            
            # Prepare format information
            if successful_tests:
                working_file = successful_tests[0]
                data_format_a = working_file['format']
                data_format_b = working_file['format']  # Assuming same format for successful transfer
            else:
                # Use the first output file's format for reporting
                data_format_a = output_files[0].suffix.lower() or 'unknown'
                data_format_b = 'unknown'
            
            # If no files worked, add a summary message
            if not is_compatible:
                compatibility_issues.insert(0, 
                    f"Script B could not process any of the {len(output_files)} output files from Script A"
                )
            
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=is_compatible,
                compatibility_issues=compatibility_issues,
                data_format_a=data_format_a,
                data_format_b=data_format_b
            )
            
        except Exception as e:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Compatibility test failed: {str(e)}"]
            )
    
    def test_pipeline_flow_with_spec(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
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
    
    def _test_pipeline_flow_with_topological_ordering(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
        """Enhanced pipeline flow testing with topological ordering and data flow chaining"""
        
        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "execution_order": [],
            "errors": []
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
                    results["errors"].append(f"No ScriptExecutionSpec found for node: {current_node}")
                    continue
                
                # Test individual script functionality first
                script_spec = script_specs[current_node]
                main_params = self.builder.get_script_main_params(script_spec)
                
                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][current_node] = script_result
                
                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {current_node} failed: {script_result.error_message}")
                    continue  # Skip data flow testing for failed scripts
                
                executed_nodes.add(current_node)
                
                # Test data compatibility with all dependent nodes
                outgoing_edges = [(src, dst) for src, dst in dag.edges if src == current_node]
                
                for src_node, dst_node in outgoing_edges:
                    if dst_node not in script_specs:
                        results["pipeline_success"] = False
                        results["errors"].append(f"Missing ScriptExecutionSpec for destination node: {dst_node}")
                        continue
                    
                    spec_a = script_specs[src_node]
                    spec_b = script_specs[dst_node]
                    
                    # Test data compatibility using enhanced matching
                    compat_result = self.test_data_compatibility_with_specs(spec_a, spec_b)
                    results["data_flow_results"][f"{src_node}->{dst_node}"] = compat_result
                    
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
    
    def _test_pipeline_flow_original(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
        """Original pipeline flow testing implementation (backward compatibility)"""
        
        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "errors": []
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
                    results["errors"].append(f"No ScriptExecutionSpec found for node: {node_name}")
                    continue
                    
                script_spec = script_specs[node_name]
                main_params = self.builder.get_script_main_params(script_spec)
                
                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][node_name] = script_result
                
                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {node_name} failed: {script_result.error_message}")
            
            # Test data flow between connected scripts using DAG edges
            for edge in dag.edges:
                script_a, script_b = edge
                
                if script_a not in script_specs or script_b not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Missing ScriptExecutionSpec for edge: {script_a} -> {script_b}")
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
            self.workspace_dir / "scripts" / f"{script_name}.py"
        ]
        
        for path in workspace_paths:
            if path.exists():
                return str(path)
        
        # Priority 3: Original fallback locations (for backward compatibility)
        fallback_paths = [
            f"src/cursus/steps/scripts/{script_name}.py",
            f"scripts/{script_name}.py",
            f"dockers/xgboost_atoz/scripts/{script_name}.py",
            f"dockers/pytorch_bsm_ext/scripts/{script_name}.py"
        ]
        
        for path in fallback_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(f"Script not found: {script_name}. Searched in workspace_dir ({self.workspace_dir}) and fallback locations.")
    
    
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
            r'.*\.tmp$',           # .tmp files
            r'.*\.temp$',          # .temp files
            r'.*~$',               # backup files ending with ~
            r'.*\.swp$',           # vim swap files
            r'.*\.bak$',           # backup files
            r'.*\.orig$',          # original files from merges
            r'.*\.rej$',           # rejected patches
            r'.*\.lock$',          # lock files
            r'.*\.pid$',           # process ID files
            r'.*\.log$',           # log files (usually not data outputs)
        ]
        
        # System files
        system_files = {
            '.ds_store',           # macOS
            'thumbs.db',           # Windows
            'desktop.ini',         # Windows
        }
        
        # Hidden files (starting with .) - but allow some exceptions
        if filename.startswith('.') and filename not in {'.gitkeep', '.placeholder'}:
            return True
            
        # Check against system files
        if filename in system_files:
            return True
            
        # Check against temp patterns
        for pattern in temp_patterns:
            if re.match(pattern, filename):
                return True
                
        # Check for cache directories and files
        if '__pycache__' in str(file_path) or filename.endswith(('.pyc', '.pyo')):
            return True
            
        return False
    
    def _find_valid_output_files(self, output_dir: Path, min_size_bytes: int = 1) -> List[Path]:
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
    
    def _generate_sample_data(self) -> Dict:
        """Generate simple sample data for testing"""
        return {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0]
        }
    
    # Phase 3: Helper Methods for Enhanced Functionality
    
    def _create_modified_spec_with_matches(self, spec_b: ScriptExecutionSpec, 
                                         path_matches: List, output_files: List[Path]) -> ScriptExecutionSpec:
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
            if hasattr(match, 'source_logical_name') and hasattr(match, 'dest_logical_name'):
                source_name = match.source_logical_name
                dest_name = match.dest_logical_name
                
                # Find matching output file
                for output_file in output_files:
                    if source_name in output_file.stem or output_file.stem in source_name:
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
            job_args=spec_b.job_args
        )
    
    def _generate_matching_report(self, path_matches: List) -> Dict[str, Any]:
        """Generate detailed matching report for debugging and analysis"""
        
        if not path_matches:
            return {
                "total_matches": 0,
                "match_types": {},
                "confidence_distribution": {},
                "recommendations": ["No logical name matches found. Consider standardizing naming conventions."]
            }
        
        # Analyze match types
        match_types = {}
        confidence_scores = []
        
        for match in path_matches:
            if hasattr(match, 'match_type') and hasattr(match, 'confidence'):
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
            recommendations.append("Average confidence is low. Consider adding more specific aliases.")
        if low_confidence > 0:
            recommendations.append(f"{low_confidence} matches have low confidence. Review naming conventions.")
        if len(match_types.get('semantic', 0)) > len(match_types.get('exact_logical', 0)):
            recommendations.append("Many semantic matches found. Consider standardizing logical names.")
        
        return {
            "total_matches": len(path_matches),
            "match_types": match_types,
            "confidence_distribution": {
                "average": avg_confidence,
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence
            },
            "recommendations": recommendations if recommendations else ["Matching looks good!"]
        }
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from file extension"""
        if not file_path or not isinstance(file_path, Path):
            return 'unknown'
        
        suffix = file_path.suffix.lower()
        
        # Map common extensions to format names
        format_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.bst': 'xgboost_model',
            '.onnx': 'onnx_model',
            '.tar.gz': 'compressed_archive',
            '.zip': 'compressed_archive',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.txt': 'text',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        
        return format_map.get(suffix, suffix[1:] if suffix else 'no_extension')
    
    # Phase 2: Logical Name Matching Methods (available when logical_name_matching module is present)
    
    def test_data_compatibility_with_logical_matching(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec):
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
                compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
            )
        
        # Find valid output files
        output_dir_a = Path(spec_a.output_paths["data_output"])
        output_files = self._find_valid_output_files(output_dir_a)
        
        if not output_files:
            return EnhancedDataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=["Script A did not produce any valid output files"]
            )
        
        # Convert to enhanced specs and use logical name matching
        enhanced_spec_a = self._convert_to_enhanced_spec(spec_a)
        enhanced_spec_b = self._convert_to_enhanced_spec(spec_b)
        
        return self.logical_name_tester.test_data_compatibility_with_logical_matching(
            enhanced_spec_a, enhanced_spec_b, output_files
        )
    
    def test_pipeline_flow_with_topological_execution(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
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
            enhanced_script_specs[node_name] = self._convert_to_enhanced_spec(script_spec)
        
        # Create script tester function
        def script_tester_func(enhanced_spec: EnhancedScriptExecutionSpec) -> ScriptTestResult:
            original_spec = self._convert_from_enhanced_spec(enhanced_spec)
            main_params = self.builder.get_script_main_params(original_spec)
            return self.test_script_with_spec(original_spec, main_params)
        
        return self.logical_name_tester.test_pipeline_with_topological_execution(
            pipeline_spec.dag, enhanced_script_specs, script_tester_func
        )
    
    def get_path_matches(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec):
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
    
    def generate_matching_report(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec) -> Dict[str, Any]:
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
    
    def validate_pipeline_logical_names(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
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
            "summary": {}
        }
        
        total_edges = 0
        valid_edges = 0
        
        for src_node, dst_node in pipeline_spec.dag.edges:
            total_edges += 1
            edge_key = f"{src_node}->{dst_node}"
            
            if src_node not in pipeline_spec.script_specs or dst_node not in pipeline_spec.script_specs:
                validation_results["edge_validations"][edge_key] = {
                    "valid": False,
                    "error": "Missing script specification"
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
                "high_confidence_matches": len([m for m in path_matches if m.confidence >= 0.8]),
                "matching_report": matching_report
            }
        
        validation_results["summary"] = {
            "total_edges": total_edges,
            "valid_edges": valid_edges,
            "validation_rate": valid_edges / total_edges if total_edges > 0 else 0.0
        }
        
        if validation_results["summary"]["validation_rate"] < 1.0:
            validation_results["recommendations"].append(
                "Some edges have no logical name matches. Consider adding aliases or standardizing naming conventions."
            )
        
        return validation_results
    
    def _convert_to_enhanced_spec(self, original_spec: ScriptExecutionSpec, 
                                input_aliases: Optional[Dict[str, List[str]]] = None,
                                output_aliases: Optional[Dict[str, List[str]]] = None):
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
            last_updated=getattr(enhanced_spec, 'last_updated', None),
            user_notes=getattr(enhanced_spec, 'user_notes', None)
        )
    
    def _generate_default_input_aliases(self, spec: ScriptExecutionSpec) -> Dict[str, List[str]]:
        """Generate default input aliases based on common patterns"""
        aliases = {}
        
        for logical_name in spec.input_paths.keys():
            alias_list = []
            
            # Common input aliases
            if "data" in logical_name.lower():
                alias_list.extend(["dataset", "input", "training_data", "processed_data"])
            if "model" in logical_name.lower():
                alias_list.extend(["artifact", "trained_model", "model_input"])
            if "config" in logical_name.lower():
                alias_list.extend(["configuration", "params", "hyperparameters", "settings"])
            
            # Add variations of the logical name
            if "_" in logical_name:
                alias_list.append(logical_name.replace("_", "-"))
                alias_list.append(logical_name.replace("_", ""))
            
            aliases[logical_name] = list(set(alias_list))  # Remove duplicates
        
        return aliases
    
    def _generate_default_output_aliases(self, spec: ScriptExecutionSpec) -> Dict[str, List[str]]:
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
