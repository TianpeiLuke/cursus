"""
Interactive Runtime Testing Factory with Config-Based Validation

This module provides the InteractiveRuntimeTestingFactory class, which transforms
the manual script testing configuration process into a guided, step-by-step workflow
for DAG-guided end-to-end testing with config-based script validation.

Enhanced Features:
- Config-based script validation (eliminates phantom scripts)
- DAG + config path input (follows PipelineDAGCompiler pattern)
- Pre-populated environment variables from config instances
- Pre-populated job arguments from config instances
- Simplified environment variable mapping (CAPITAL_CASE rules)
- Integration with enhanced ScriptAutoDiscovery

Preserved Features:
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
from .config_aware_script_resolver import ConfigAwareScriptPathResolver


class InteractiveRuntimeTestingFactory:
    """
    Enhanced interactive factory with config-based script validation.
    
    NEW FEATURES:
    - ✅ Config-based script validation (eliminates phantom scripts)
    - ✅ DAG + config path input (follows PipelineDAGCompiler pattern)
    - ✅ Pre-populated environment variables from config instances
    - ✅ Pre-populated job arguments from config instances
    - ✅ Simplified environment variable mapping (CAPITAL_CASE rules)
    - ✅ Integration with enhanced ScriptAutoDiscovery
    
    PRESERVED FEATURES:
    - ✅ DAG-guided script discovery and analysis
    - ✅ Step-by-step interactive configuration
    - ✅ Immediate validation with detailed feedback
    - ✅ Auto-configuration for eligible scripts
    - ✅ Complete end-to-end testing orchestration
    
    Enhanced Example Usage:
        >>> dag = create_xgboost_complete_e2e_dag()
        >>> config_path = "pipeline_config/config_NA_xgboost_AtoZ_v2/config_NA_xgboost_AtoZ.json"
        >>> factory = InteractiveRuntimeTestingFactory(dag, config_path)
        >>> 
        >>> # Only discovers validated scripts (no phantoms)
        >>> scripts_to_test = factory.get_scripts_requiring_testing()
        >>> 
        >>> # Environment variables and job arguments pre-populated from config
        >>> for script_name in factory.get_pending_script_configurations():
        >>>     requirements = factory.get_script_testing_requirements(script_name)
        >>>     # requirements['environment_variables'] populated from config
        >>>     # requirements['job_arguments'] populated from config
        >>>     
        >>>     factory.configure_script_testing(
        >>>         script_name,
        >>>         expected_inputs={'data_input': 'path/to/input'},
        >>>         expected_outputs={'data_output': 'path/to/output'}
        >>>         # environment_variables automatically from config!
        >>>         # job_arguments automatically from config!
        >>>     )
        >>> 
        >>> results = factory.execute_dag_guided_testing()
    """
    
    def __init__(self, dag: PipelineDAG, config_path: Optional[str] = None, workspace_dir: str = "test/integration/runtime"):
        """
        Initialize factory with optional config path for enhanced validation.
        
        Args:
            dag: Pipeline DAG to analyze and test
            config_path: Optional path to pipeline configuration JSON file
            workspace_dir: Workspace directory for testing files
        """
        self.dag = dag
        self.config_path = config_path
        self.workspace_dir = Path(workspace_dir)
        
        # Enhanced state management with config integration
        self.script_specs: Dict[str, ScriptExecutionSpec] = {}
        self.script_info_cache: Dict[str, Dict[str, Any]] = {}
        self.pending_scripts: List[str] = []
        self.auto_configured_scripts: List[str] = []
        
        # Config-based components (None if no config provided)
        self.loaded_configs: Optional[Dict[str, Any]] = None
        self.script_resolver: Optional[ConfigAwareScriptPathResolver] = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            # Enhanced config-based workflow
            self._initialize_with_config()
        else:
            # Legacy DAG-only workflow with phantom script warnings
            self._initialize_legacy_mode()
            self.logger.warning("⚠️ Using legacy DAG-only mode - phantom scripts may be discovered")
            self.logger.warning("💡 Consider providing config_path for enhanced validation")
        
        self.logger.info(f"✅ Initialized InteractiveRuntimeTestingFactory with {len(self.script_info_cache)} validated scripts")
    
    def _initialize_with_config(self):
        """Initialize with config-based validation (enhanced mode)."""
        try:
            # Load configs using existing utilities
            from ...steps.configs.utils import load_configs, build_complete_config_classes
            from ...step_catalog.adapters.config_resolver import StepConfigResolverAdapter
            
            # Load and filter configs to DAG-related only
            config_classes = build_complete_config_classes()
            all_configs = load_configs(self.config_path, config_classes)
            
            config_resolver = StepConfigResolverAdapter()
            dag_nodes = list(self.dag.nodes)
            self.loaded_configs = config_resolver.resolve_config_map(
                dag_nodes=dag_nodes,
                available_configs=all_configs
            )
            
            # Initialize unified script resolver (Phase 2 enhancement)
            self.script_resolver = ConfigAwareScriptPathResolver()
            
            # Discover and analyze scripts using unified resolution
            self._discover_and_analyze_scripts_from_config()
            
            self.logger.info(f"✅ Config-based initialization completed with {len(self.loaded_configs)} configs")
            
        except Exception as e:
            self.logger.error(f"❌ Config-based initialization failed: {e}")
            self.logger.warning("🔄 Falling back to legacy DAG-only mode")
            self._initialize_legacy_mode()
    
    def _initialize_legacy_mode(self):
        """Initialize with legacy DAG-only mode (backward compatibility)."""
        # Use existing infrastructure directly
        self.spec_builder = PipelineTestingSpecBuilder(
            test_data_dir=str(self.workspace_dir),
            step_catalog=self._initialize_step_catalog()
        )
        
        # Discover and analyze scripts using existing logic
        self._discover_and_analyze_scripts()
    
    def _initialize_step_catalog(self) -> StepCatalog:
        """Initialize step catalog with unified workspace resolution."""
        workspace_dirs = [self.workspace_dir]
        return StepCatalog(workspace_dirs=workspace_dirs)
    
    def _discover_and_analyze_scripts_from_config(self) -> None:
        """
        Enhanced script discovery using unified ConfigAwareScriptPathResolver - eliminates phantom scripts!
        
        This method uses the unified script resolver with config instances to provide
        definitive script validation, eliminating phantom scripts by only discovering
        scripts that have actual entry points defined in config instances.
        """
        # Use unified resolver with config instances - eliminates phantom scripts!
        for node_name in self.dag.nodes:
            if node_name not in self.loaded_configs:
                self.logger.debug(f"No config found for DAG node: {node_name}")
                continue
            
            config_instance = self.loaded_configs[node_name]
            
            # Use unified resolver to get script path (eliminates phantom scripts)
            script_path = self.script_resolver.resolve_script_path(config_instance)
            
            if not script_path:
                self.logger.debug(f"Skipping {node_name}: no script entry point (phantom script eliminated)")
                continue  # Skip configs without scripts - no phantom scripts!
            
            # Get config validation info for enhanced metadata
            validation_info = self.script_resolver.validate_config_for_script_resolution(config_instance)
            
            # Extract environment variables and job arguments from config
            config_environ_vars = self._extract_environ_vars_from_config(config_instance)
            config_job_args = self._extract_job_args_from_config(config_instance)
            
            # Cache enhanced script information with config metadata
            self.script_info_cache[node_name] = {
                'script_name': node_name,
                'step_name': node_name,
                'script_path': script_path,  # RELIABLE PATH FROM UNIFIED RESOLVER
                'expected_inputs': ['data_input'],  # Still need user input
                'expected_outputs': ['data_output'],  # Still need user input
                'default_input_paths': {'data_input': f"test/data/{node_name}/input"},
                'default_output_paths': {'data_output': f"test/data/{node_name}/output"},
                'config_environ_vars': config_environ_vars,  # From config!
                'config_job_args': config_job_args,  # From config!
                'source_dir': validation_info.get('source_dir'),
                'entry_point_field': validation_info.get('entry_point'),
                'entry_point_value': validation_info.get('entry_point'),
                'config_type': validation_info.get('config_type'),
                'auto_configurable': self._can_auto_configure_from_config_type(validation_info.get('config_type', '')),
                # Legacy compatibility fields
                'default_environ_vars': config_environ_vars or {'CURSUS_ENV': 'testing'},
                'default_job_args': config_job_args or {'job_type': 'testing'}
            }
            
            # Determine configuration status
            if self.script_info_cache[node_name]['auto_configurable']:
                self.auto_configured_scripts.append(node_name)
            else:
                self.pending_scripts.append(node_name)
        
        self.logger.info(f"📊 Unified Script Discovery Summary:")
        self.logger.info(f"   - Validated scripts: {len(self.script_info_cache)} (phantom scripts eliminated)")
        self.logger.info(f"   - Auto-configurable: {len(self.auto_configured_scripts)} scripts")
        self.logger.info(f"   - Pending configuration: {len(self.pending_scripts)} scripts")
    
    def _can_auto_configure_from_config_type(self, config_type: str) -> bool:
        """Check if script can be auto-configured based on config type."""
        # Scripts like Package, Registration, Payload can be auto-configured
        auto_configurable_types = ['PackageConfig', 'RegistrationConfig', 'PayloadConfig']
        return any(config_type.endswith(auto_type) for auto_type in auto_configurable_types)
    
    def _extract_environ_vars_from_config(self, config_instance) -> Dict[str, str]:
        """Extract environment variables from config instance."""
        environ_vars = {}
        
        # Check common environment variable fields
        env_fields = [
            'environment_variables',
            'environ_vars', 
            'env_vars',
            'runtime_environment'
        ]
        
        for field in env_fields:
            if hasattr(config_instance, field):
                field_value = getattr(config_instance, field)
                if isinstance(field_value, dict):
                    environ_vars.update(field_value)
                    break
        
        # Add framework-specific environment variables if available
        if hasattr(config_instance, 'framework_version'):
            framework_version = getattr(config_instance, 'framework_version')
            if framework_version:
                environ_vars['FRAMEWORK_VERSION'] = str(framework_version)
        
        return environ_vars
    
    def _extract_job_args_from_config(self, config_instance) -> Dict[str, str]:
        """Extract job arguments from config instance."""
        job_args = {}
        
        # Check common job argument fields
        job_arg_fields = [
            'job_arguments',
            'job_args',
            'hyperparameters',
            'algorithm_specification'
        ]
        
        for field in job_arg_fields:
            if hasattr(config_instance, field):
                field_value = getattr(config_instance, field)
                if isinstance(field_value, dict):
                    # Convert all values to strings for job arguments
                    job_args.update({k: str(v) for k, v in field_value.items()})
                    break
        
        # Add job type if available
        if hasattr(config_instance, 'job_type'):
            job_type = getattr(config_instance, 'job_type')
            if job_type:
                job_args['job_type'] = str(job_type)
        
        return job_args
    
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
        Get enhanced requirements with config-populated defaults.
        
        This method provides detailed information about what inputs, outputs,
        environment variables, and job arguments are needed for testing a script,
        with enhanced config-based defaults and source indicators.
        
        Args:
            script_name: Name of the script to get requirements for
            
        Returns:
            Dictionary containing detailed requirements information with config sources
            
        Raises:
            ValueError: If script_name is not found in validated scripts
        """
        if script_name not in self.script_info_cache:
            raise ValueError(f"Script '{script_name}' not found in validated scripts")
        
        info = self.script_info_cache[script_name]
        
        # Check if we have config-based environment variables and job arguments
        has_config_environ_vars = 'config_environ_vars' in info and info['config_environ_vars']
        has_config_job_args = 'config_job_args' in info and info['config_job_args']
        
        # Build environment variables with source indicators
        environment_variables = []
        if has_config_environ_vars:
            # Config-based environment variables (enhanced mode)
            for name, value in info['config_environ_vars'].items():
                environment_variables.append({
                    'name': name,
                    'description': f"Environment variable: {name}",
                    'required': False,
                    'default_value': value,
                    'source': 'config'  # NEW: Indicates value comes from config
                })
        else:
            # Legacy environment variables (backward compatibility)
            for name, value in info['default_environ_vars'].items():
                environment_variables.append({
                    'name': name,
                    'description': f"Environment variable: {name}",
                    'required': False,
                    'default_value': value,
                    'source': 'legacy'  # Indicates legacy default
                })
        
        # Build job arguments with source indicators
        job_arguments = []
        if has_config_job_args:
            # Config-based job arguments (enhanced mode)
            for name, value in info['config_job_args'].items():
                job_arguments.append({
                    'name': name,
                    'description': f"Job argument: {name}",
                    'required': False,
                    'default_value': value,
                    'source': 'config'  # NEW: Indicates value comes from config
                })
        else:
            # Legacy job arguments (backward compatibility)
            for name, value in info['default_job_args'].items():
                job_arguments.append({
                    'name': name,
                    'description': f"Job argument: {name}",
                    'required': False,
                    'default_value': value,
                    'source': 'legacy'  # Indicates legacy default
                })
        
        result = {
            'script_name': info['script_name'],
            'step_name': info['step_name'],
            'script_path': info['script_path'],
            'source_dir': info.get('source_dir'),
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
            'environment_variables': environment_variables,
            'job_arguments': job_arguments,
            'auto_configurable': info.get('auto_configurable', script_name in self.auto_configured_scripts)
        }
        
        # Add config metadata if available (enhanced mode)
        if 'config_type' in info:
            result['config_metadata'] = {
                'entry_point_field': info.get('entry_point_field'),
                'entry_point_value': info.get('entry_point_value'),
                'config_type': info.get('config_type'),
                'workspace_id': info.get('workspace_id')
            }
        
        return result
    
    # === INTERACTIVE CONFIGURATION ===
    
    def configure_script_testing(self, script_name: str, **kwargs) -> ScriptExecutionSpec:
        """
        Enhanced configuration with config-populated defaults.
        
        Users only need to provide expected_inputs and expected_outputs.
        Environment variables and job arguments are pre-populated from config.
        
        Args:
            script_name: Name of the script to configure
            **kwargs: Configuration parameters including:
                - expected_inputs or input_paths: Dict mapping input names to file paths
                - expected_outputs or output_paths: Dict mapping output names to file paths
                - environment_variables or environ_vars: Dict of environment variables (optional, uses config defaults)
                - job_arguments or job_args: Dict of job arguments (optional, uses config defaults)
                
        Returns:
            Configured ScriptExecutionSpec object
            
        Raises:
            ValueError: If script_name is not found or configuration validation fails
            
        Enhanced Example:
            >>> # Config-based workflow - environment variables automatically populated!
            >>> factory.configure_script_testing(
            ...     'tabular_preprocessing',
            ...     expected_inputs={'data_input': 'test/data/input.csv'},
            ...     expected_outputs={'data_output': 'test/output/processed.csv'}
            ...     # environment_variables automatically from config!
            ...     # job_arguments automatically from config!
            ... )
            >>> 
            >>> # User can override config defaults if needed
            >>> factory.configure_script_testing(
            ...     'xgboost_training',
            ...     expected_inputs={'training_data': 'test/data/train.csv'},
            ...     expected_outputs={'model': 'test/output/model.pkl'},
            ...     environment_variables={'FRAMEWORK_VERSION': '2.0.0'}  # Override config default
            ... )
        """
        if script_name not in self.script_info_cache:
            raise ValueError(f"Script '{script_name}' not found in validated scripts")
        
        info = self.script_info_cache[script_name]
        
        # Extract configuration inputs with flexible parameter names
        input_paths = kwargs.get('expected_inputs', kwargs.get('input_paths', {}))
        output_paths = kwargs.get('expected_outputs', kwargs.get('output_paths', {}))
        
        # Use config defaults for environment variables and job arguments (user can override)
        # Priority: user override > config defaults > legacy defaults
        user_environ_vars = kwargs.get('environment_variables', kwargs.get('environ_vars', {}))
        user_job_args = kwargs.get('job_arguments', kwargs.get('job_args', {}))
        
        # Start with config defaults if available, otherwise use legacy defaults
        if 'config_environ_vars' in info and info['config_environ_vars']:
            # Enhanced mode: use config-populated environment variables
            environ_vars = info['config_environ_vars'].copy()
            environ_vars.update(user_environ_vars)  # User overrides take precedence
        else:
            # Legacy mode: use legacy defaults
            environ_vars = info['default_environ_vars'].copy()
            environ_vars.update(user_environ_vars)  # User overrides take precedence
        
        if 'config_job_args' in info and info['config_job_args']:
            # Enhanced mode: use config-populated job arguments
            job_args = info['config_job_args'].copy()
            job_args.update(user_job_args)  # User overrides take precedence
        else:
            # Legacy mode: use legacy defaults
            job_args = info['default_job_args'].copy()
            job_args.update(user_job_args)  # User overrides take precedence
        
        # Immediate validation with detailed feedback
        validation_errors = self._validate_script_configuration(info, input_paths, output_paths)
        
        if validation_errors:
            raise ValueError(f"Configuration validation failed for {script_name}:\n" + 
                           "\n".join(f"  - {error}" for error in validation_errors))
        
        # Create ScriptExecutionSpec with config-enhanced data
        script_spec = ScriptExecutionSpec(
            script_name=script_name,
            step_name=info['step_name'],
            script_path=info['script_path'],
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,  # From config with user overrides
            job_args=job_args,  # From config with user overrides
            last_updated=datetime.now().isoformat(),
            user_notes=f"Configured with config-populated defaults from {self.config_path or 'legacy mode'}"
        )
        
        # Store configuration and update state
        self.script_specs[script_name] = script_spec
        if script_name in self.pending_scripts:
            self.pending_scripts.remove(script_name)
        
        # Log configuration details
        config_source = "config" if 'config_environ_vars' in info and info['config_environ_vars'] else "legacy"
        user_overrides = len(user_environ_vars) + len(user_job_args)
        self.logger.info(f"✅ {script_name} configured successfully with {config_source} defaults" + 
                        (f" and {user_overrides} user overrides" if user_overrides > 0 else ""))
        
        return script_spec
    
    def _validate_script_configuration(self, info: Dict[str, Any], input_paths: Dict[str, str], 
                                     output_paths: Dict[str, str]) -> List[str]:
        """
        Enhanced validation with config-aware error messages.
        
        Args:
            info: Cached script information
            input_paths: Dictionary of input name to file path mappings
            output_paths: Dictionary of output name to file path mappings
            
        Returns:
            List of validation error messages (empty if valid)
        """
        validation_errors = []
        
        # Enhanced validation for required inputs with config context
        for input_name in info['expected_inputs']:
            if input_name not in input_paths:
                error_msg = f"Missing required input: {input_name}"
                if 'config_type' in info:
                    error_msg += f" (script from {info['config_type']})"
                if 'entry_point_field' in info:
                    error_msg += f" [entry point: {info['entry_point_field']}]"
                validation_errors.append(error_msg)
            else:
                input_path = input_paths[input_name]
                if not Path(input_path).exists():
                    error_msg = f"Input file does not exist: {input_path}"
                    if 'source_dir' in info and info['source_dir']:
                        error_msg += f" (expected in workspace: {info['source_dir']})"
                    validation_errors.append(error_msg)
                elif Path(input_path).stat().st_size == 0:
                    validation_errors.append(f"Input file is empty: {input_path}")
        
        # Enhanced validation for required outputs with config context
        for output_name in info['expected_outputs']:
            if output_name not in output_paths:
                error_msg = f"Missing required output: {output_name}"
                if 'config_type' in info:
                    error_msg += f" (script from {info['config_type']})"
                validation_errors.append(error_msg)
        
        # Config-specific validation hints
        if validation_errors and 'config_type' in info:
            config_hint = f"💡 Config source: {self.config_path or 'unknown'}"
            if 'entry_point_value' in info:
                config_hint += f" | Entry point: {info['entry_point_value']}"
            validation_errors.append(config_hint)
        
        return validation_errors
    
    # === END-TO-END TESTING EXECUTION ===
    
    def execute_dag_guided_testing(self) -> Dict[str, Any]:
        """
        Enhanced DAG-guided testing with config metadata in results.
        
        This method orchestrates the complete testing process, ensuring all scripts
        are configured and then executing the full pipeline testing using the
        existing RuntimeTester infrastructure, with enhanced config metadata for traceability.
        
        Returns:
            Dictionary containing comprehensive testing results with enhanced config metadata
            
        Raises:
            ValueError: If there are scripts that still need configuration
            
        Enhanced Example:
            >>> results = factory.execute_dag_guided_testing()
            >>> print(f"Config mode: {results['config_integration_metadata']['mode']}")
            >>> print(f"Config automation: {results['config_integration_metadata']['automation_percentage']:.1f}%")
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
        
        # Enhanced config integration analysis for results
        config_mode = "enhanced" if self.config_path else "legacy"
        scripts_with_config_defaults = 0
        scripts_with_config_env_vars = 0
        scripts_with_config_job_args = 0
        
        for info in self.script_info_cache.values():
            if 'config_environ_vars' in info and info['config_environ_vars']:
                scripts_with_config_defaults += 1
                scripts_with_config_env_vars += 1
            if 'config_job_args' in info and info['config_job_args']:
                scripts_with_config_job_args += 1
        
        # Enhance results with comprehensive config metadata for traceability
        results["config_integration_metadata"] = {
            "mode": config_mode,
            "config_path": self.config_path,
            "config_loaded": self.loaded_configs is not None,
            "total_loaded_configs": len(self.loaded_configs) if self.loaded_configs else 0,
            "phantom_elimination_active": config_mode == "enhanced",
            "automation_percentage": (scripts_with_config_defaults / len(self.script_info_cache) * 100) if self.script_info_cache else 0,
            "scripts_with_config_env_vars": scripts_with_config_env_vars,
            "scripts_with_config_job_args": scripts_with_config_job_args,
            "testing_timestamp": datetime.now().isoformat()
        }
        
        # Enhanced interactive factory information with config details
        results["interactive_factory_info"] = {
            "dag_name": getattr(self.dag, 'name', 'unnamed'),
            "total_scripts": len(self.script_info_cache),
            "auto_configured_scripts": len(self.auto_configured_scripts),
            "manually_configured_scripts": len(self.script_specs) - len(self.auto_configured_scripts),
            "script_configurations": {
                name: {
                    "auto_configured": name in self.auto_configured_scripts,
                    "step_name": self.script_info_cache[name]['step_name'],
                    # NEW: Config source metadata for each script
                    "config_type": self.script_info_cache[name].get('config_type'),
                    "has_config_env_vars": bool(self.script_info_cache[name].get('config_environ_vars')),
                    "has_config_job_args": bool(self.script_info_cache[name].get('config_job_args')),
                    "entry_point_field": self.script_info_cache[name].get('entry_point_field'),
                    "workspace_id": self.script_info_cache[name].get('workspace_id')
                }
                for name in self.script_specs.keys()
            }
        }
        
        # Log enhanced completion details
        config_details = f"with {config_mode} mode"
        if config_mode == "enhanced":
            automation_pct = results["config_integration_metadata"]["automation_percentage"]
            config_details += f" ({automation_pct:.1f}% config automation)"
        
        self.logger.info(f"✅ DAG-guided testing completed for {len(self.script_specs)} scripts {config_details}")
        return results
    
    # === FACTORY STATUS AND SUMMARY ===
    
    def get_testing_factory_summary(self) -> Dict[str, Any]:
        """
        Enhanced summary with config integration status and detailed metadata.
        
        This method provides a complete overview of the factory's current state,
        including script counts, configuration status, config integration info,
        and detailed script information with config source indicators.
        
        Returns:
            Dictionary containing comprehensive factory status information with config integration details
            
        Enhanced Example:
            >>> summary = factory.get_testing_factory_summary()
            >>> print(f"Config mode: {summary['config_integration']['mode']}")
            >>> print(f"Config source: {summary['config_integration']['config_path']}")
            >>> print(f"Scripts with config defaults: {summary['config_integration']['scripts_with_config_defaults']}")
            >>> print(f"Ready for testing: {summary['ready_for_testing']}")
            >>> print(f"Completion: {summary['completion_percentage']:.1f}%")
        """
        total_scripts = len(self.script_info_cache)
        configured_scripts = len(self.script_specs)
        auto_configured_scripts = len(self.auto_configured_scripts)
        manually_configured_scripts = configured_scripts - auto_configured_scripts
        pending_scripts = len(self.pending_scripts)
        
        # Enhanced config integration analysis
        config_mode = "enhanced" if self.config_path else "legacy"
        scripts_with_config_defaults = 0
        scripts_with_config_env_vars = 0
        scripts_with_config_job_args = 0
        
        for info in self.script_info_cache.values():
            if 'config_environ_vars' in info and info['config_environ_vars']:
                scripts_with_config_defaults += 1
                scripts_with_config_env_vars += 1
            if 'config_job_args' in info and info['config_job_args']:
                scripts_with_config_job_args += 1
        
        # Build enhanced summary with config integration details
        summary = {
            'dag_name': getattr(self.dag, 'name', 'unnamed'),
            'total_scripts': total_scripts,
            'configured_scripts': configured_scripts,
            'auto_configured_scripts': auto_configured_scripts,
            'manually_configured_scripts': manually_configured_scripts,
            'pending_scripts': pending_scripts,
            'ready_for_testing': pending_scripts == 0,
            'completion_percentage': (configured_scripts / total_scripts * 100) if total_scripts > 0 else 0,
            
            # NEW: Config integration status
            'config_integration': {
                'mode': config_mode,
                'config_path': self.config_path,
                'config_loaded': self.loaded_configs is not None,
                'total_loaded_configs': len(self.loaded_configs) if self.loaded_configs else 0,
                'scripts_with_config_defaults': scripts_with_config_defaults,
                'scripts_with_config_env_vars': scripts_with_config_env_vars,
                'scripts_with_config_job_args': scripts_with_config_job_args,
                'phantom_elimination_active': config_mode == "enhanced"
            },
            
            # Enhanced script details with config source information
            'script_details': {
                name: {
                    'status': 'auto_configured' if name in self.auto_configured_scripts 
                             else 'configured' if name in self.script_specs 
                             else 'pending',
                    'step_name': info['step_name'],
                    'expected_inputs': len(info['expected_inputs']),
                    'expected_outputs': len(info['expected_outputs']),
                    # NEW: Config source indicators
                    'config_type': info.get('config_type'),
                    'has_config_env_vars': bool(info.get('config_environ_vars')),
                    'has_config_job_args': bool(info.get('config_job_args')),
                    'entry_point_field': info.get('entry_point_field'),
                    'workspace_id': info.get('workspace_id')
                }
                for name, info in self.script_info_cache.items()
            }
        }
        
        # Add config efficiency metrics
        if total_scripts > 0:
            summary['config_integration']['config_automation_percentage'] = (
                scripts_with_config_defaults / total_scripts * 100
            )
        else:
            summary['config_integration']['config_automation_percentage'] = 0
        
        return summary
    
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
