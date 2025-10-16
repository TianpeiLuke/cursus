"""
Interactive Runtime Testing Factory

This module provides the InteractiveRuntimeTestingFactory class, which transforms
the manual script testing configuration process into a guided, step-by-step workflow
for DAG-guided end-to-end testing.

Key Features:
- DAG-guided script discovery and analysis
- Step-by-step interactive configuration
- Immediate validation with detailed feedback
- Auto-configuration for eligible scripts
- Complete end-to-end testing orchestration
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ...api.dag.base_dag import PipelineDAG
from ...step_catalog import StepCatalog
from .runtime_models import ScriptExecutionSpec, PipelineTestingSpec
from .runtime_spec_builder import PipelineTestingSpecBuilder
from .runtime_testing import RuntimeTester


class InteractiveRuntimeTestingFactory:
    """
    Interactive factory for DAG-guided script runtime testing.
    
    This factory transforms manual script testing configuration into a guided,
    step-by-step workflow that provides intelligent script discovery, interactive
    user input collection, and automated testing orchestration.
    
    Features:
    - ✅ DAG-guided script discovery and analysis
    - ✅ Step-by-step interactive configuration
    - ✅ Immediate validation with detailed feedback
    - ✅ Auto-configuration for eligible scripts
    - ✅ Complete end-to-end testing orchestration
    - ❌ No complex requirements extraction (uses existing contract discovery)
    - ❌ No elaborate data models (uses existing ScriptExecutionSpec)
    - ❌ No framework-specific analysis (uses existing step catalog)
    
    Example Usage:
        >>> dag = create_xgboost_complete_e2e_dag()
        >>> factory = InteractiveRuntimeTestingFactory(dag)
        >>> 
        >>> # 1. Automatic script discovery and analysis
        >>> scripts_to_test = factory.get_scripts_requiring_testing()
        >>> summary = factory.get_testing_factory_summary()
        >>> 
        >>> # 2. Step-by-step interactive configuration
        >>> for script_name in factory.get_pending_script_configurations():
        >>>     requirements = factory.get_script_testing_requirements(script_name)
        >>>     factory.configure_script_testing(
        >>>         script_name,
        >>>         expected_inputs={'data_input': 'path/to/input'},
        >>>         expected_outputs={'data_output': 'path/to/output'}
        >>>     )
        >>> 
        >>> # 3. Complete end-to-end testing execution
        >>> results = factory.execute_dag_guided_testing()
    """
    
    def __init__(self, dag: PipelineDAG, workspace_dir: str = "test/integration/runtime"):
        """
        Initialize Interactive Runtime Testing Factory with DAG analysis.
        
        Args:
            dag: Pipeline DAG to analyze and test
            workspace_dir: Workspace directory for testing files and configurations
        """
        self.dag = dag
        self.workspace_dir = Path(workspace_dir)
        
        # Use existing infrastructure directly
        self.spec_builder = PipelineTestingSpecBuilder(
            test_data_dir=workspace_dir,
            step_catalog=self._initialize_step_catalog()
        )
        
        # Simplified state management
        self.script_specs: Dict[str, ScriptExecutionSpec] = {}
        self.pending_scripts: List[str] = []
        self.auto_configured_scripts: List[str] = []
        self.script_info_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Discover and analyze scripts using existing logic
        self._discover_and_analyze_scripts()
        
        self.logger.info(f"✅ Initialized InteractiveRuntimeTestingFactory for DAG with {len(self.dag.nodes)} scripts")
    
    def _initialize_step_catalog(self) -> StepCatalog:
        """Initialize step catalog with unified workspace resolution."""
        workspace_dirs = [self.workspace_dir]
        return StepCatalog(workspace_dirs=workspace_dirs)
    
    # === DAG-GUIDED SCRIPT DISCOVERY ===
    
    def _discover_and_analyze_scripts(self) -> None:
        """
        DAG-guided script discovery using existing PipelineTestingSpecBuilder intelligence.
        
        This method leverages the existing intelligent script resolution capabilities
        to discover scripts from the DAG and cache their information for interactive guidance.
        """
        for node_name in self.dag.nodes:
            try:
                # Use existing intelligent resolution with step catalog
                script_spec = self.spec_builder._resolve_script_with_step_catalog_if_available(node_name)
                
                if not script_spec:
                    # Fallback to existing intelligent resolution
                    script_spec = self.spec_builder.resolve_script_execution_spec_from_node(node_name)
                
                # Cache script information for interactive guidance
                self.script_info_cache[script_spec.script_name] = {
                    'script_name': script_spec.script_name,
                    'step_name': script_spec.step_name,
                    'script_path': script_spec.script_path,
                    'expected_inputs': list(script_spec.input_paths.keys()),
                    'expected_outputs': list(script_spec.output_paths.keys()),
                    'default_input_paths': script_spec.input_paths.copy(),
                    'default_output_paths': script_spec.output_paths.copy(),
                    'default_environ_vars': script_spec.environ_vars.copy(),
                    'default_job_args': script_spec.job_args.copy()
                }
                
                # Check if script can be auto-configured
                if self._can_auto_configure(script_spec):
                    self.script_specs[script_spec.script_name] = script_spec
                    self.auto_configured_scripts.append(script_spec.script_name)
                else:
                    # Needs user configuration
                    self.pending_scripts.append(script_spec.script_name)
                    
            except Exception as e:
                self.logger.warning(f"Could not resolve script for node {node_name}: {e}")
                # Add to pending for manual configuration
                self.pending_scripts.append(node_name)
                self._add_fallback_script_info(node_name)
        
        self.logger.info(f"📊 Script Discovery Summary:")
        self.logger.info(f"   - Auto-configured: {len(self.auto_configured_scripts)} scripts")
        self.logger.info(f"   - Pending configuration: {len(self.pending_scripts)} scripts")
    
    def _can_auto_configure(self, spec: ScriptExecutionSpec) -> bool:
        """
        Check if script can be auto-configured (input files exist).
        
        Args:
            spec: Script execution specification to check
            
        Returns:
            True if all input files exist and script can be auto-configured
        """
        for input_path in spec.input_paths.values():
            if not Path(input_path).exists():
                return False
        return True
    
    def _add_fallback_script_info(self, node_name: str) -> None:
        """
        Add fallback script info for unknown scripts.
        
        Args:
            node_name: Name of the DAG node that couldn't be resolved
        """
        self.script_info_cache[node_name] = {
            'script_name': node_name,
            'step_name': node_name,
            'script_path': f"scripts/{node_name}.py",
            'expected_inputs': ['data_input'],
            'expected_outputs': ['data_output'],
            'default_input_paths': {'data_input': f"test/data/{node_name}/input"},
            'default_output_paths': {'data_output': f"test/data/{node_name}/output"},
            'default_environ_vars': {'CURSUS_ENV': 'testing'},
            'default_job_args': {'job_type': 'testing'}
        }
    
    # === INTERACTIVE WORKFLOW METHODS ===
    
    def get_scripts_requiring_testing(self) -> List[str]:
        """
        Get all scripts discovered from DAG that need testing configuration.
        
        Returns:
            List of script names that were discovered from the DAG
        """
        return list(self.script_info_cache.keys())
    
    def get_pending_script_configurations(self) -> List[str]:
        """
        Get scripts that still need user configuration.
        
        Returns:
            List of script names that require manual configuration
        """
        return self.pending_scripts.copy()
    
    def get_auto_configured_scripts(self) -> List[str]:
        """
        Get scripts that were auto-configured.
        
        Returns:
            List of script names that were automatically configured
        """
        return self.auto_configured_scripts.copy()
    
    def get_script_testing_requirements(self, script_name: str) -> Dict[str, Any]:
        """
        Get interactive requirements for testing a specific script.
        
        This method provides detailed information about what inputs, outputs,
        environment variables, and job arguments are needed for testing a script.
        
        Args:
            script_name: Name of the script to get requirements for
            
        Returns:
            Dictionary containing detailed requirements information with examples
            
        Raises:
            ValueError: If script_name is not found in discovered scripts
        """
        if script_name not in self.script_info_cache:
            raise ValueError(f"Script '{script_name}' not found in discovered scripts")
        
        info = self.script_info_cache[script_name]
        
        return {
            'script_name': info['script_name'],
            'step_name': info['step_name'],
            'script_path': info['script_path'],
            'expected_inputs': [
                {
                    'name': name,
                    'description': f"Input data for {name}",
                    'required': True,
                    'example_path': f"test/data/{script_name}/input/{name}",
                    'current_path': info['default_input_paths'].get(name, '')
                }
                for name in info['expected_inputs']
            ],
            'expected_outputs': [
                {
                    'name': name,
                    'description': f"Output data for {name}",
                    'required': True,
                    'example_path': f"test/data/{script_name}/output/{name}",
                    'current_path': info['default_output_paths'].get(name, '')
                }
                for name in info['expected_outputs']
            ],
            'environment_variables': [
                {
                    'name': name,
                    'description': f"Environment variable: {name}",
                    'required': False,
                    'default_value': value
                }
                for name, value in info['default_environ_vars'].items()
            ],
            'job_arguments': [
                {
                    'name': name,
                    'description': f"Job argument: {name}",
                    'required': False,
                    'default_value': value
                }
                for name, value in info['default_job_args'].items()
            ],
            'auto_configurable': script_name in self.auto_configured_scripts
        }
    
    # === INTERACTIVE CONFIGURATION ===
    
    def configure_script_testing(self, script_name: str, **kwargs) -> ScriptExecutionSpec:
        """
        Configure testing for a script with immediate validation.
        
        This method allows users to configure script testing parameters with
        immediate validation and detailed feedback on any configuration issues.
        
        Args:
            script_name: Name of the script to configure
            **kwargs: Configuration parameters including:
                - expected_inputs or input_paths: Dict mapping input names to file paths
                - expected_outputs or output_paths: Dict mapping output names to file paths
                - environment_variables or environ_vars: Dict of environment variables
                - job_arguments or job_args: Dict of job arguments
                
        Returns:
            Configured ScriptExecutionSpec object
            
        Raises:
            ValueError: If script_name is not found or configuration validation fails
            
        Example:
            >>> factory.configure_script_testing(
            ...     'data_preprocessing',
            ...     expected_inputs={'raw_data': 'test/data/raw.csv'},
            ...     expected_outputs={'processed_data': 'test/output/processed.csv'}
            ... )
        """
        if script_name not in self.script_info_cache:
            raise ValueError(f"Script '{script_name}' not found in discovered scripts")
        
        info = self.script_info_cache[script_name]
        
        # Extract configuration inputs with flexible parameter names
        input_paths = kwargs.get('expected_inputs', kwargs.get('input_paths', {}))
        output_paths = kwargs.get('expected_outputs', kwargs.get('output_paths', {}))
        environ_vars = kwargs.get('environment_variables', kwargs.get('environ_vars', info['default_environ_vars']))
        job_args = kwargs.get('job_arguments', kwargs.get('job_args', info['default_job_args']))
        
        # Immediate validation with detailed feedback
        validation_errors = self._validate_script_configuration(info, input_paths, output_paths)
        
        if validation_errors:
            raise ValueError(f"Configuration validation failed for {script_name}:\n" + 
                           "\n".join(f"  - {error}" for error in validation_errors))
        
        # Create ScriptExecutionSpec using existing model
        script_spec = ScriptExecutionSpec(
            script_name=script_name,
            step_name=info['step_name'],
            script_path=info['script_path'],
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=job_args,
            last_updated=datetime.now().isoformat(),
            user_notes=f"Configured by InteractiveRuntimeTestingFactory"
        )
        
        # Store configuration and update state
        self.script_specs[script_name] = script_spec
        if script_name in self.pending_scripts:
            self.pending_scripts.remove(script_name)
        
        self.logger.info(f"✅ {script_name} configured successfully for testing")
        return script_spec
    
    def _validate_script_configuration(self, info: Dict[str, Any], input_paths: Dict[str, str], 
                                     output_paths: Dict[str, str]) -> List[str]:
        """
        Validate script configuration with detailed feedback.
        
        Args:
            info: Cached script information
            input_paths: Dictionary of input name to file path mappings
            output_paths: Dictionary of output name to file path mappings
            
        Returns:
            List of validation error messages (empty if valid)
        """
        validation_errors = []
        
        # Validate required inputs
        for input_name in info['expected_inputs']:
            if input_name not in input_paths:
                validation_errors.append(f"Missing required input: {input_name}")
            else:
                input_path = input_paths[input_name]
                if not Path(input_path).exists():
                    validation_errors.append(f"Input file does not exist: {input_path}")
                elif Path(input_path).stat().st_size == 0:
                    validation_errors.append(f"Input file is empty: {input_path}")
        
        # Validate required outputs
        for output_name in info['expected_outputs']:
            if output_name not in output_paths:
                validation_errors.append(f"Missing required output: {output_name}")
        
        return validation_errors
    
    # === END-TO-END TESTING EXECUTION ===
    
    def execute_dag_guided_testing(self) -> Dict[str, Any]:
        """
        Execute comprehensive DAG-guided end-to-end testing.
        
        This method orchestrates the complete testing process, ensuring all scripts
        are configured and then executing the full pipeline testing using the
        existing RuntimeTester infrastructure.
        
        Returns:
            Dictionary containing comprehensive testing results with factory context
            
        Raises:
            ValueError: If there are scripts that still need configuration
            
        Example:
            >>> results = factory.execute_dag_guided_testing()
            >>> print(f"Tested {results['interactive_factory_info']['total_scripts']} scripts")
        """
        # Check that all scripts are configured
        if self.pending_scripts:
            pending_info = []
            for script_name in self.pending_scripts:
                requirements = self.get_script_testing_requirements(script_name)
                pending_info.append(f"  - {script_name}: needs {len(requirements['expected_inputs'])} inputs")
            
            raise ValueError(
                f"Cannot execute testing - missing configuration for {len(self.pending_scripts)} scripts:\n" +
                "\n".join(pending_info) +
                f"\n\nUse factory.configure_script_testing(script_name, expected_inputs={{...}}, expected_outputs={{...}}) to configure each script."
            )
        
        # Execute comprehensive testing using existing infrastructure
        pipeline_spec = PipelineTestingSpec(
            dag=self.dag,
            script_specs=self.script_specs,
            test_workspace_root=str(self.workspace_dir)
        )
        
        tester = RuntimeTester(
            config_or_workspace_dir=str(self.workspace_dir),
            step_catalog=StepCatalog(workspace_dirs=[self.workspace_dir])
        )
        
        # Execute enhanced testing
        results = tester.test_pipeline_flow_with_step_catalog_enhancements(pipeline_spec)
        
        # Enhance results with interactive factory information
        results["interactive_factory_info"] = {
            "dag_name": getattr(self.dag, 'name', 'unnamed'),
            "total_scripts": len(self.script_info_cache),
            "auto_configured_scripts": len(self.auto_configured_scripts),
            "manually_configured_scripts": len(self.script_specs) - len(self.auto_configured_scripts),
            "script_configurations": {
                name: {
                    "auto_configured": name in self.auto_configured_scripts,
                    "step_name": self.script_info_cache[name]['step_name']
                }
                for name in self.script_specs.keys()
            }
        }
        
        self.logger.info(f"✅ DAG-guided testing completed for {len(self.script_specs)} scripts")
        return results
    
    # === FACTORY STATUS AND SUMMARY ===
    
    def get_testing_factory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of interactive testing factory state.
        
        This method provides a complete overview of the factory's current state,
        including script counts, configuration status, and detailed script information.
        
        Returns:
            Dictionary containing comprehensive factory status information
            
        Example:
            >>> summary = factory.get_testing_factory_summary()
            >>> print(f"Ready for testing: {summary['ready_for_testing']}")
            >>> print(f"Completion: {summary['completion_percentage']:.1f}%")
        """
        total_scripts = len(self.script_info_cache)
        configured_scripts = len(self.script_specs)
        auto_configured_scripts = len(self.auto_configured_scripts)
        manually_configured_scripts = configured_scripts - auto_configured_scripts
        pending_scripts = len(self.pending_scripts)
        
        return {
            'dag_name': getattr(self.dag, 'name', 'unnamed'),
            'total_scripts': total_scripts,
            'configured_scripts': configured_scripts,
            'auto_configured_scripts': auto_configured_scripts,
            'manually_configured_scripts': manually_configured_scripts,
            'pending_scripts': pending_scripts,
            'ready_for_testing': pending_scripts == 0,
            'completion_percentage': (configured_scripts / total_scripts * 100) if total_scripts > 0 else 0,
            'script_details': {
                name: {
                    'status': 'auto_configured' if name in self.auto_configured_scripts 
                             else 'configured' if name in self.script_specs 
                             else 'pending',
                    'step_name': info['step_name'],
                    'expected_inputs': len(info['expected_inputs']),
                    'expected_outputs': len(info['expected_outputs'])
                }
                for name, info in self.script_info_cache.items()
            }
        }
    
    # === UTILITY METHODS ===
    
    def validate_configuration_preview(self, script_name: str, input_paths: Dict[str, str]) -> List[str]:
        """
        Preview validation issues without configuring the script.
        
        This utility method allows users to check for configuration issues
        before actually configuring the script.
        
        Args:
            script_name: Name of the script to validate
            input_paths: Dictionary of input name to file path mappings
            
        Returns:
            List of validation issues (empty if valid)
            
        Raises:
            ValueError: If script_name is not found in discovered scripts
        """
        if script_name not in self.script_info_cache:
            raise ValueError(f"Script '{script_name}' not found in discovered scripts")
        
        info = self.script_info_cache[script_name]
        issues = []
        
        for name, path in input_paths.items():
            if not Path(path).exists():
                issues.append(f"Input file missing: {name} -> {path}")
            elif Path(path).stat().st_size == 0:
                issues.append(f"Input file empty: {name} -> {path}")
        
        return issues
    
    def get_script_info(self, script_name: str) -> Dict[str, Any]:
        """
        Get basic script information for user guidance.
        
        Args:
            script_name: Name of the script to get information for
            
        Returns:
            Dictionary containing basic script information
            
        Raises:
            ValueError: If script_name is not found in discovered scripts
        """
        if script_name not in self.script_info_cache:
            raise ValueError(f"Script '{script_name}' not found in discovered scripts")
        
        info = self.script_info_cache[script_name]
        
        return {
            'script_name': info['script_name'],
            'script_path': info['script_path'],
            'step_name': info['step_name'],
            'expected_inputs': info['expected_inputs'],
            'expected_outputs': info['expected_outputs'],
            'example_input_paths': {
                name: f"test/data/{script_name}/input/{name}"
                for name in info['expected_inputs']
            },
            'example_output_paths': {
                name: f"test/data/{script_name}/output/{name}"
                for name in info['expected_outputs']
            },
            'auto_configurable': script_name in self.auto_configured_scripts
        }
