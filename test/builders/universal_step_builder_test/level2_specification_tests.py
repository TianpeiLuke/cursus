"""
Level 2 Specification Tests for step builders.

These tests focus on the correct usage of specifications and contracts:
- Specification presence and structure
- Contract alignment with specification
- Environment variable handling
"""

from .base_test import UniversalStepBuilderTestBase


class SpecificationTests(UniversalStepBuilderTestBase):
    """
    Level 2 tests focusing on specification compliance.
    
    These tests validate that a step builder correctly uses specifications
    and aligns with script contracts. These tests require an understanding
    of the specification system but not deep knowledge of property paths
    or dependency resolution.
    """
    
    def test_specification_usage(self) -> None:
        """Test that the builder uses a valid specification."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Check that spec is available
        self._assert(
            hasattr(builder, 'spec'),
            f"Builder must have a spec attribute"
        )
        
        self._assert(
            builder.spec is not None,
            f"Builder must have a non-None spec attribute"
        )
        
        # Verify spec has required attributes
        required_spec_attrs = [
            'step_type',
            'node_type',
            'dependencies',
            'outputs'
        ]
        
        for attr in required_spec_attrs:
            self._assert(
                hasattr(builder.spec, attr),
                f"Specification must have {attr} attribute"
            )
    
    def test_contract_alignment(self) -> None:
        """Test that the specification aligns with the script contract."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Check contract is available
        self._assert(
            hasattr(builder, 'contract'),
            f"Builder must have a contract attribute"
        )
        
        if builder.contract is None:
            self._log("Contract is None, skipping contract alignment tests")
            return
        
        # Verify contract has required attributes
        required_contract_attrs = [
            'entry_point',
            'expected_input_paths',
            'expected_output_paths'
        ]
        
        for attr in required_contract_attrs:
            self._assert(
                hasattr(builder.contract, attr),
                f"Contract must have {attr} attribute"
            )
        
        # Verify all dependency logical names have corresponding paths in contract
        if hasattr(builder.spec, 'dependencies'):
            for dep_name, dep_spec in builder.spec.dependencies.items():
                logical_name = dep_spec.logical_name
                if logical_name != "hyperparameters_s3_uri":  # Special case
                    self._assert(
                        logical_name in builder.contract.expected_input_paths,
                        f"Dependency {logical_name} must have corresponding path in contract"
                    )
        
        # Verify all output logical names have corresponding paths in contract
        if hasattr(builder.spec, 'outputs'):
            for out_name, out_spec in builder.spec.outputs.items():
                logical_name = out_spec.logical_name
                self._assert(
                    logical_name in builder.contract.expected_output_paths,
                    f"Output {logical_name} must have corresponding path in contract"
                )
    
    def test_environment_variable_handling(self) -> None:
        """Test that the builder handles environment variables correctly."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Get environment variables
        env_vars = builder._get_environment_variables()
        
        self._assert(
            isinstance(env_vars, dict),
            f"_get_environment_variables() must return a dictionary"
        )
        
        # Verify environment variables include required variables from contract
        if hasattr(builder, 'contract') and builder.contract is not None:
            if hasattr(builder.contract, 'required_env_vars'):
                for env_var in builder.contract.required_env_vars:
                    self._assert(
                        env_var in env_vars,
                        f"Environment variables must include required variable {env_var}"
                    )
            
            # Verify environment variables include optional variables with defaults
            if hasattr(builder.contract, 'optional_env_vars'):
                for env_var, default in builder.contract.optional_env_vars.items():
                    self._assert(
                        env_var in env_vars,
                        f"Environment variables must include optional variable {env_var}"
                    )
                    
    def test_job_arguments(self) -> None:
        """Test that the builder correctly generates job arguments from script contract."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Skip if no contract
        if not hasattr(builder, 'contract') or builder.contract is None:
            self._log("Contract is None, skipping job arguments test")
            return
            
        # Get job arguments
        job_args = builder._get_job_arguments()
        
        # If contract has no expected arguments, job_args should be None
        if not hasattr(builder.contract, 'expected_arguments') or not builder.contract.expected_arguments:
            self._assert(
                job_args is None,
                "_get_job_arguments() should return None when contract has no expected arguments"
            )
            return
            
        # Otherwise, job_args should be a list
        self._assert(
            isinstance(job_args, list),
            f"_get_job_arguments() must return a list or None, got {type(job_args)}"
        )
        
        # Check that all expected arguments are included
        for arg_name, arg_value in builder.contract.expected_arguments.items():
            arg_flag = f"--{arg_name}"
            
            # Check that the flag is in the list
            self._assert(
                arg_flag in job_args,
                f"Job arguments must include flag {arg_flag}"
            )
            
            # Check that the value follows the flag
            if arg_flag in job_args:
                flag_index = job_args.index(arg_flag)
                
                # Ensure there's a value after the flag
                self._assert(
                    flag_index + 1 < len(job_args),
                    f"Job argument {arg_flag} must be followed by a value"
                )
                
                # Check that the value is correct
                if flag_index + 1 < len(job_args):
                    self._assert(
                        job_args[flag_index + 1] == arg_value,
                        f"Job argument {arg_flag} has value {job_args[flag_index + 1]}, expected {arg_value}"
                    )
